import re
from io import BytesIO
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import base64

from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules


# KONFIGURASI APLIKASI #


st.set_page_config(
    page_title="Sistem Analisis Asosiasi Obat BPJS Kesehatan PRB",
    layout="wide"
)

# Header identitas apotek
logo_apotek = ["logo_apotek.jpg"]

logo_path = None
for candidate in logo_apotek:
    if Path(candidate).exists():
        logo_path = candidate
        break

header_col1, header_col2 = st.columns([1.1, 8.9], vertical_alignment="center")

with header_col1:
    if logo_path is not None:
        st.image(logo_path, width=140)

with header_col2:
    st.markdown(
        """
        <div style="display: flex; flex-direction: column; justify-content: center;">
            <div style="font-size: 26px; font-weight: 700; margin-bottom: 10px;">
                Apotek Rajendra
            </div>
            <div style="font-size: 42px; font-weight: 800; line-height: 1.15;">
                Sistem Analisis Asosiasi Obat BPJS Kesehatan PRB
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div style="margin-top: 22px; margin-bottom: 8px; font-size: 16px;">
        Upload data resep dari Farmalite untuk menghasilkan aturan asosiasi obat.
        File yang diunggah harus berisi kolom <b>No. Fraktur</b> dan <b>Produk</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# FUNGSI BANTUAN #


def trim_upper(x):
    x = "" if pd.isna(x) else str(x)
    x = x.upper()
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def trim_text(x):
    x = "" if pd.isna(x) else str(x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def regex_replace(text, pattern, repl=""):
    return re.sub(pattern, repl, str(text))


def contains(text, keyword):
    return keyword in str(text)


def baca_mapping(path="Mapping nama obat.xlsx"):
    mapping = pd.read_excel(path)
    mapping = mapping.iloc[:, :2].copy()
    mapping.columns = ["raw", "master"]

    mapping["raw"] = mapping["raw"].astype(str).str.upper().str.strip()
    mapping["raw"] = mapping["raw"].apply(lambda x: re.sub(r"\s+", " ", x))

    mapping["master"] = mapping["master"].astype(str).str.strip()
    mapping["master"] = mapping["master"].apply(lambda x: re.sub(r"\s+", " ", x))

    return dict(zip(mapping["raw"], mapping["master"]))



# PREPROCESSING #


def proses_preprocessing(df):
    df = df.copy()

    # Ambil dan rename kolom awal
    df = df.rename(columns={
        "No. Fraktur": "ID Resep Asli",
        "Produk": "Produk"
    })

    df = df[["ID Resep Asli", "Produk"]].copy()
    df = df.dropna(subset=["ID Resep Asli", "Produk"])

    # 1. Standarisasi ID Resep: R00001, R00002, dst
    unique_ids = df["ID Resep Asli"].drop_duplicates().tolist()
    id_map = {
        old_id: f"R{i+1:05d}"
        for i, old_id in enumerate(unique_ids)
    }

    df["ID Resep"] = df["ID Resep Asli"].map(id_map)

    # Kolom B: Produk
    df["B_Produk"] = df["Produk"].astype(str)

    # 1. Replace nama obat khusus
    replace_produk_awal = {
        "BPJS PRB GLIKLAZID TAB 80 MG 100S DEXA":
            "BPJS PRB GLICLAZIDE TAB 80 MG 100S DEXA",

        "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HCL) TAB 2 MG 100S MERSI":
            "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HYDROCHLORIDE) TAB 2 MG 100S MERSI",

        "BPJS PRB FONYLIN MR (GLIKLAZIDE) TAB 60MG 30S FERRON PAR PHARMACEUTICALS":
            "BPJS PRB FONYLIN MR (GLICLAZIDE) TAB 60MG 30S FERRON PAR PHARMACEUTICALS",

        "BPJS PRB HYDROCHLOROTHIAZIDE (HCT) TAB 25 MG 100S KIMIA FARMA":
            "BPJS PRB HYDROCHLOROTHIAZIDE (HYDROCHLOROTHIAZIDE) TAB 25 MG 100S KIMIA FARMA",

        "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HCL) TAB 2 MG 100S HOLIPHARMA":
            "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HYDROCHLORIDE) TAB 2 MG 100S HOLIPHARMA",

        "BPJS PRB PROPANOLOL 10MG OGB DEXA":
            "BPJS PRB PROPRANOLOL 10MG OGB DEXA",

        "BPJS PRB RETAPHYL SR (THEOPHYLINE)":
            "BPJS PRB RETAPHYL SR (THEOPHYLLINE)",

        "BPJS PRB NOVOMIX 30 (ASPART) 100 IU/ML 5S 3ML INSULIN":
            "BPJS PRB NOVOMIX 30 FLEXPEN (BIPHASIC INSULIN ASPART) 100 IU/ML 5S 3ML INSULIN",

        "BPJS PRB GLIKLAZIDE MR 60 MG PRATAPA 30 S":
            "BPJS PRB GLICLAZIDE MR 60 MG PRATAPA 30S",

        "BPJS PRB HYDROCHLOROTHIAZIDE (HCT) KIMIA FARMA 25 MG 100S":
            "BPJS PRB HYDROCHLOROTHIAZIDE (HYDROCHLOROTHIAZIDE) KIMIA FARMA 25 MG 100S",

        "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HCL) MERSI TAB 2 MG 100S":
            "BPJS PRB HEXYMER (TRIHEXYPHENIDYL HYDROCHLORIDE) MERSI TAB 2 MG 100S",

        "BPJS PRB GLIKLAZID DEXA 80 MG 100S":
            "BPJS PRB GLICLAZIDE DEXA 80 MG 100S",

        "BPJS PRB FONYLIN MR (GLIKLAZIDE) FERRON PAR PHARMACEUTICALS 60MG 30S":
            "BPJS PRB FONYLIN MR (GLICLAZIDE) FERRON PAR PHARMACEUTICALS 60MG 30S",

        "BPJS PRB RETAPHYL SR (THEOPHYLINE) 300 Mg":
            "BPJS PRB RETAPHYL SR (THEOPHYLLINE) 300 Mg",

        "BPJS PRB TRIHEXYPHENIDYL HCL HOLI 2MG TAB 100S":
            "BPJS PRB TRIHEXYPHENIDYL HYDROCHLORIDE HOLI 2MG TAB 100S",
    }

    # Normalisasi key agar tetap cocok walaupun kapital/spasi berbeda
    replace_produk_awal = {
        trim_upper(k): v for k, v in replace_produk_awal.items()
    }

    df["B_Produk_Replace"] = df["B_Produk"].apply(
        lambda x: replace_produk_awal.get(trim_upper(x), x)
    )

    # Hapus BPJS PRB
    df["C_tanpa_bpjs_prb"] = df["B_Produk_Replace"].apply(
        lambda x: trim_upper(str(x).replace("BPJS PRB ", ""))
    )

    # Hapus merek dagang
    def cleaning_d(x):
        hasil = x
        daftar_hapus = [
            " TAB",
            " HJ",
            " DEXA",
            " MEGA",
            " KF",
            " BETA",
            " KIMIA FARMA",
            " 100S",
            " 30S",
            " 200S"
        ]
        for item in daftar_hapus:
            hasil = hasil.replace(item, "")
        return trim_text(hasil)

    df["D_tanpa_merk"] = df["C_tanpa_bpjs_prb"].apply(cleaning_d)

    # Ambil isi tanda kurung kalau ada
    def cleaning_e(x):
        x = str(x)
        if "(" in x and ")" in x and x.find("(") < x.find(")"):
            inside = x[x.find("(")+1:x.find(")")]
            after = x[x.find(")")+1:]
            return trim_text(inside + " " + after)
        return trim_text(x)

    df["E_cleaning_1"] = df["D_tanpa_merk"].apply(cleaning_e)

    # Hapus merek yang masih lolos
    def cleaning_f(x):
        pattern = r"5S|1S|TURBUHALER|DISKUS|DOSIS|INHALATION|INHALER|CAPS|INSULIN|BAYER|YARINDO|MITSUBISHI|FERRON PAR PHARMACEUTICALS"
        return trim_text(regex_replace(x, pattern, ""))

    df["F_cleaning_2"] = df["E_cleaning_1"].apply(cleaning_f)

    # Hapus jumlah strip
    def cleaning_g(x):
        x = str(x).replace(",", ".")
        x = regex_replace(x, r"\b(BESILATE|FUMARATE)\b", "")
        x = regex_replace(x, r"\s*/\s*", " + ")
        x = regex_replace(x, r"\s+(60|120|200)\b\s*$", "")
        x = regex_replace(x, r"\s{2,}", " ")
        return trim_text(x)

    df["G_cleaning_3"] = df["F_cleaning_2"].apply(cleaning_g)

    # Hapus merek yang masih lolos
    def cleaning_h(x):
        x = regex_replace(x, r"\b\d+\s*S\b", "")
        x = regex_replace(x, r"\b(TEMPO SCAN|MERSI|IKAPHARMINDO|IMFARMIND)\b", "")
        return trim_text(x)

    df["H_cleaning_4"] = df["G_cleaning_3"].apply(cleaning_h)

    # Hapus LET 1
    df["I_hapus_let_1"] = df["H_cleaning_4"].apply(
        lambda x: trim_text(regex_replace(x, r"(MG|MCG|ML)LET\b", r"\1"))
    )

    # Hapus LET 2
    def cleaning_j(x):
        x = trim_text(x)
        if x == "PHENOBARBITALLET":
            return "PHENOBARBITAL"
        if x == "VILDAGLIPTINLET":
            return "VILDAGLIPTIN"
        return x

    df["J_hapus_let_2"] = df["I_hapus_let_1"].apply(cleaning_j)

    # Hapus obat tanpa dosis
    def cleaning_l(x):
        daftar = ["PIOGLITAZONE", "SPIRONOLACTONE", "HALOPERIDOL", "ATORVASTATIN"]
        ada_obat = any(re.search(rf"^{obat}\b", str(x)) for obat in daftar)
        ada_angka = bool(re.search(r"\d", str(x)))
        if ada_obat and not ada_angka:
            return "HAPUS (TANPA DOSIS)"
        return "OK"

    df["L_hapus_obat_sistem_lama"] = df["J_hapus_let_2"].apply(cleaning_l)

    # Cek dosis
    df["M_cek_dosis"] = df["J_hapus_let_2"].apply(
        lambda x: "MASIH TANPA DOSIS" if not bool(re.search(r"\d", str(x))) else "AMAN"
    )

    # Kasih dosis
    def cleaning_n(x):
        x = trim_text(x)

        mapping_dosis = {
            "CLOPIDOGREL": "CLOPIDOGREL 75 MG",
            "FUROSEMIDE": "FUROSEMIDE 40 MG",
            "CARVEDILOL": "CARVEDILOL 6.25 MG",
            "CEPEZET": "CEPEZET 100 MG",
            "VILDAGLIPTIN": "VILDAGLIPTIN 50 MG",
            "RETAPHYL SR": "THEOPHYLLINE 300 MG",
            "PHENOBARBITAL": "PHENOBARBITAL 30 MG",
            "VITAMIN B1 MARIN LIZA": "VITAMIN B1 50 MG",
            "SALBULIN": "SALBULIN INHALER",
            "RYZODEG FLEX TOUCH": "RYZODEG FLEX TOUCH",
            "THEOPHYLLINE": "THEOPHYLLINE 300 MG",
            "ATORVASTATIN MEDIKA": "ATORVASTATIN 20 MG",
            "INDACETROL": "INDACETROL INHALER"
        }

        return mapping_dosis.get(x, x)

    df["N_kasih_dosis"] = df["J_hapus_let_2"].apply(cleaning_n)

    # Obat hasil cleaning 1
    df["P_obat_awal"] = df["N_kasih_dosis"]

    # Hapus merk yang masih lolos
    def cleaning_q(x):
        x = str(x).upper()

        # Kasih spasi antara angka dan satuan
        x = regex_replace(x, r"(\d)(MG|MCG)\b", r"\1 \2")

        pattern_merk = (
            r"\b(DARYA-VARIA|PIM|SAMCO|SAMPHARINDO|SAMPARINDO|PHARMACON|"
            r"KALBE|FARMA|MEPRO|PRATAPA|NIRMALA|NOVAPHARIN|INDOFARMA|"
            r"PRATPA|NIRMALA|HOLI PHARMA|HOLIPHARMA|OGB|ESA|MARIN|LIZA|"
            r"NOVEL|NOVA|HOLI|ACTAVIS|ETA|MEDICA|MEDIKA|DEXA|MEGA|HJ|KF|"
            r"BETA|KIMIA)\b"
        )

        x = regex_replace(x, pattern_merk, "")
        x = regex_replace(x, r"\s{2,}", " ")
        return trim_text(x)

    df["Q_hapus_merk"] = df["P_obat_awal"].apply(cleaning_q)

    # amlodipin 10 mg 10 mg
    def cleaning_r(x):
        x = trim_text(x)
        if x == "AMLODIPINE 10 MG 10 MG":
            return "AMLODIPINE 10 MG"
        if x == "DIGOXIN 0.25 0.25 MG":
            return "DIGOXIN 0.25 MG"
        return x

    df["R_amlodipin_dobel"] = df["Q_hapus_merk"].apply(cleaning_r)

    # ASETOSAL 80 MG 2
    df["S_asetosal_80_mg_2"] = df["R_amlodipin_dobel"].apply(
        lambda x: "ASETOSAL 80 MG" if trim_text(x) == "ASETOSAL 80 MG 2" else x
    )

    # FENOTEROL 100 MCG 200 100 MCG
    df["T_fenoterol"] = df["S_asetosal_80_mg_2"].apply(
        lambda x: "FENOTEROL HYDROBROMIDE 100 MCG"
        if trim_text(x) == "FENOTEROL HYDROBROMIDE 100 MCG 200 100 MCG"
        else x
    )

    # ISOSORBIDE DINITRATE NIRMALA10 MG
    df["U_nirmala10"] = df["T_fenoterol"].apply(
        lambda x: str(x).replace("NIRMALA10", "NIRMALA 10")
        if "NIRMALA10" in str(x)
        else x
    )

    # HAPUS NIRMALA
    df["V_hapus_nirmala"] = df["U_nirmala10"].apply(
        lambda x: trim_text(str(x).replace("NIRMALA ", ""))
        if "NIRMALA 10" in str(x)
        else x
    )

    # HAPUS SULFATE
    df["W_hapus_sulfate"] = df["V_hapus_nirmala"].apply(
        lambda x: trim_text(regex_replace(x, r"SALBUTAMOL\s+SULFATE", "SALBUTAMOL"))
    )

    # HAPUS SYR
    def cleaning_x(x):
        x = trim_text(x)
        if x == "VALPROIC ACID SYR 250 MG + 5ML":
            return "VALPROIC ACID 250 MG + 5ML 100ML"
        return x

    df["X_hapus_syr"] = df["W_hapus_sulfate"].apply(cleaning_x)

    # ganti adalat oros
    def cleaning_y(x):
        x = trim_text(x)
        if x == "ADALAT OROS 30 MG":
            return "NIFEDIPINE 30 MG (OROS)"
        return x

    df["Y_ganti_adalat_oros"] = df["X_hapus_syr"].apply(cleaning_y)

    # ganti acetosal
    def cleaning_z(x):
        x = trim_text(x)
        if "ASETOSAL" in x or "ACETYLSALICYLIC ACID" in x:
            return "ACETOSAL 80 MG"
        return x

    df["Z_ganti_acetosal"] = df["Y_ganti_adalat_oros"].apply(cleaning_z)

    # MR gliclazide
    def cleaning_aa(x):
        x = trim_text(x)

        # Samakan ejaan GLIKLAZIDE menjadi GLICLAZIDE
        x = x.replace("GLIKLAZIDE", "GLICLAZIDE")

        # Pastikan gliclazide 60 mg menjadi bentuk MR
        x = x.replace("GLICLAZIDE 60 MG", "GLICLAZIDE MR 60 MG")

        return trim_text(x)

    df["AA_mr_gliclazide"] = df["Z_ganti_acetosal"].apply(cleaning_aa)

    # Lispro tambah mix
    def cleaning_ab(row):
        produk_awal = str(row["B_Produk"]).upper()
        hasil = row["AA_mr_gliclazide"]

        if "MIX 25" in produk_awal:
            return hasil + " MIX 25"
        if "MIX 50" in produk_awal:
            return hasil + " MIX 50"
        return hasil

    df["AB_lispro_tambah_mix"] = df.apply(cleaning_ab, axis=1)

    # hapus angka akhir di theophylline
    def cleaning_ac(x):
        x = trim_text(x)
        if "THEOPHYLLINE" in x:
            return trim_text(regex_replace(x, r" \d+$", ""))
        return x

    df["AC_hapus_2_theo"] = df["AB_lispro_tambah_mix"].apply(cleaning_ac)

    # ganti budesonide
    df["AD_ganti_budesonide"] = df["AC_hapus_2_theo"].apply(
        lambda x: str(x).replace(
            "BUDESONIDE + 160 + 4.5",
            "BUDESONIDE + FORMOTEROL 160 + 4.5 MCG"
        )
    )

    # Mapping nama obat
    kamus_mapping = baca_mapping()
    df["Nama Standar"] = df["AD_ganti_budesonide"].map(kamus_mapping)

    # Kalau ada yang tidak ketemu mapping, tetap ditampilkan agar bisa dicek
    df["Nama Standar"] = df["Nama Standar"].fillna(df["AD_ganti_budesonide"])
    df["Nama Standar"] = df["Nama Standar"].apply(trim_text)

    # Data bersih sebelum duplikasi
    df_bersih = df[["ID Resep", "Nama Standar"]].copy()
    df_bersih = df_bersih.rename(columns={"Nama Standar": "Nama Obat"})

    # Rapikan teks dan hapus baris yang kemungkinan merupakan header ikut terbaca sebagai data
    df_bersih["ID Resep"] = df_bersih["ID Resep"].astype(str).apply(trim_text)
    df_bersih["Nama Obat"] = df_bersih["Nama Obat"].astype(str).apply(trim_text)

    df_bersih = df_bersih[
        ~(
            df_bersih["ID Resep"].str.upper().eq("ID RESEP")
            | df_bersih["Nama Obat"].str.upper().eq("NAMA OBAT")
            | df_bersih["ID Resep"].str.upper().str.contains("ID RESEP", na=False)
            | df_bersih["Nama Obat"].str.upper().str.contains("NAMA OBAT", na=False)
        )
    ]

    df_bersih = df_bersih[
        (df_bersih["ID Resep"] != "")
        & (df_bersih["Nama Obat"] != "")
    ]

    # Hapus duplikasi ID Resep + Nama Obat
    df_bersih = df_bersih.drop_duplicates(subset=["ID Resep", "Nama Obat"])

    # Transformasi data menjadi transaksi
    transaksi = (
        df_bersih
        .groupby("ID Resep")["Nama Obat"]
        .apply(lambda x: "; ".join(pd.unique(x)))
        .reset_index()
    )

    transaksi.columns = ["ID Resep", "Nama Obat"]

    return df, df_bersih, transaksi


# FP-GROWTH DAN ASSOCIATION RULES #


def proses_arm(transaksi, min_support=0.026, min_confidence=0.20):
    list_transaksi = transaksi["Nama Obat"].apply(
        lambda x: [item.strip() for item in str(x).split(";") if item.strip() != ""]
    ).tolist()

    te = TransactionEncoder()
    te_array = te.fit(list_transaksi).transform(list_transaksi)

    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    frequent_itemsets = fpgrowth(
        df_encoded,
        min_support=min_support,
        use_colnames=True
    )

    if frequent_itemsets.empty:
        return frequent_itemsets, pd.DataFrame()

    rules = association_rules(
        frequent_itemsets,
        metric="confidence",
        min_threshold=min_confidence
    )

    rules = rules[rules["lift"] > 1].copy()
    rules = rules.sort_values(by="confidence", ascending=False).reset_index(drop=True)

    return frequent_itemsets, rules


def format_rules_for_display(rules):
    tampil = rules.copy()

    tampil["antecedents"] = tampil["antecedents"].apply(lambda x: ", ".join(list(x)))
    tampil["consequents"] = tampil["consequents"].apply(lambda x: ", ".join(list(x)))

    kolom = ["antecedents", "consequents", "support", "confidence", "lift"]
    tampil = tampil[kolom]

    tampil = tampil.rename(columns={
        "antecedents": "Obat Pemicu (Antecedent)",
        "consequents": "Obat yang Ikut Muncul (Consequent)",
        "support": "Support",
        "confidence": "Confidence",
        "lift": "Lift"
    })

    tampil["Support"] = tampil["Support"].round(2)
    tampil["Confidence"] = tampil["Confidence"].round(2)
    tampil["Lift"] = tampil["Lift"].round(2)

    return tampil


def hitung_obat_unik_dari_transaksi(transaksi):
    list_obat_transaksi = []

    for daftar_obat in transaksi["Nama Obat"]:
        obat_per_resep = [
            obat.strip()
            for obat in str(daftar_obat).split(";")
            if obat.strip() != ""
            and obat.strip().upper() != "NAMA OBAT"
            and obat.strip().upper() != "ID RESEP"
        ]
        list_obat_transaksi.extend(obat_per_resep)

    daftar_obat_unik = sorted(set(list_obat_transaksi))
    return daftar_obat_unik


def format_desimal(nilai, digit=1):
    teks = f"{nilai:.{digit}f}"
    teks = teks.rstrip("0").rstrip(".")
    teks = teks.replace(".", ",")
    return teks


def format_persen(nilai):
    # Untuk interpretasi card: 1 angka di belakang koma
    return format_desimal(nilai * 100, digit=1) + "%"


def format_angka(nilai):
    # Untuk interpretasi card: 1 angka di belakang koma
    return format_desimal(nilai, digit=1)


def format_bilangan(nilai):
    return f"{int(nilai):,}".replace(",", ".")


def buat_interpretasi_rules(rules, total_resep, jumlah=3):
    rules_top = rules.head(jumlah).copy()
    hasil = []

    for i, row in rules_top.iterrows():
        antecedent = ", ".join(list(row["antecedents"]))
        consequent = ", ".join(list(row["consequents"]))

        support = format_persen(row["support"])
        confidence = format_persen(row["confidence"])
        lift = format_angka(row["lift"])

        jumlah_kombinasi = round(row["support"] * total_resep)
        jumlah_antecedent = round(row["antecedent support"] * total_resep)
        jumlah_confidence = round(row["confidence"] * jumlah_antecedent)

        hasil.append({
            "nomor": i + 1,
            "antecedent": antecedent,
            "consequent": consequent,
            "support": support,
            "confidence": confidence,
            "lift": lift,
            "jumlah_kombinasi": jumlah_kombinasi,
            "jumlah_antecedent": jumlah_antecedent,
            "jumlah_confidence": jumlah_confidence
        })

    return hasil



# INFORMASI KEWASPADAAN OBAT #


VARIASI_KEKUATAN_ACUAN = {
    "acarbose": "50 mg dan 100 mg",
    "acetylsalicylic acid": "80 mg dan 100 mg",
    "amlodipine": "5 mg dan 10 mg",
    "atorvastatin": "10 mg dan 20 mg",
    "bisoprolol": "1,25 mg; 2,5 mg; dan 5 mg",
    "budesonide + formoterol": "80/4,5 mcg dan 160/4,5 mcg",
    "candesartan": "8 mg dan 16 mg",
    "captopril": "12,5 mg; 25 mg; dan 50 mg",
    "carvedilol": "6,25 mg dan 25 mg",
    "diltiazem": "30 mg; 100 mg; dan 200 mg",
    "divalproex sodium": "250 mg dan 500 mg",
    "gliclazide": "80 mg dan 60 mg",
    "glimepiride": "1 mg; 2 mg; 3 mg; dan 4 mg",
    "glyceryl trinitrate": "2,5 mg dan 5 mg",
    "irbesartan": "150 mg dan 300 mg",
    "isosorbide dinitrate": "5 mg dan 10 mg",
    "lisinopril": "5 mg dan 10 mg",
    "metformin": "500 mg dan 850 mg",
    "nifedipine": "10 mg dan 30 mg",
    "pioglitazone": "15 mg dan 30 mg",
    "ramipril": "2,5 mg; 5 mg; dan 10 mg",
    "salmeterol + fluticasone": "25/50 mcg; 50/100 mcg; 50/250 mcg; dan 50/500 mcg",
    "simvastatin": "10 mg dan 20 mg",
    "spironolactone": "25 mg dan 100 mg",
    "telmisartan": "40 mg dan 80 mg",
    "theophylline": "150 mg dan 300 mg",
    "valsartan": "80 mg dan 160 mg"
}


PASANGAN_ZAT_AKTIF_MIRIP = [
    ("amiodarone", "amantadine"),
    ("amlodipine", "amiloride"),
    ("atorvastatin", "atomoxetine"),
    ("captopril", "carvedilol"),
    ("chlorpromazine", "chlordiazepoxide"),
    ("chlorpromazine", "chlorpropamide"),
    ("clonidine", "clonazepam"),
    ("clonidine", "clozapine"),
    ("clozapine", "clonazepam"),
    ("diltiazem", "diazepam"),
    ("hydrochlorothiazide", "hydralazine"),
    ("hydrochlorothiazide", "hydroxyzine"),
    ("hydrochlorothiazide", "hydroxychloroquine"),
    ("metformin", "metronidazole"),
    ("nifedipine", "nicardipine"),
    ("nifedipine", "nimodipine"),
    ("phenobarbital", "pentobarbital"),
    ("risperidone", "ropinirole")
]


def buat_tabel_variasi_kekuatan_acuan():
    return pd.DataFrame(
        [
            {
                "Zat Aktif": zat_aktif,
                "Kekuatan Sediaan": kekuatan
            }
            for zat_aktif, kekuatan in VARIASI_KEKUATAN_ACUAN.items()
        ]
    )


def buat_tabel_zat_aktif_mirip_acuan():
    return pd.DataFrame(
        PASANGAN_ZAT_AKTIF_MIRIP,
        columns=[
            "Nama Zat Aktif",
            "Pasangan Nama Zat Aktif yang Mirip"
        ]
    )


PASANGAN_MIRIP_MAP = {}
for zat_aktif_1, zat_aktif_2 in PASANGAN_ZAT_AKTIF_MIRIP:
    PASANGAN_MIRIP_MAP.setdefault(zat_aktif_1, set()).add(zat_aktif_2)
    PASANGAN_MIRIP_MAP.setdefault(zat_aktif_2, set()).add(zat_aktif_1)


DAFTAR_ZAT_AKTIF_ACUAN = sorted(
    set(VARIASI_KEKUATAN_ACUAN.keys()) | set(PASANGAN_MIRIP_MAP.keys()),
    key=len,
    reverse=True
)


def normalisasi_nama_obat(nama_obat):
    nama_obat = "" if pd.isna(nama_obat) else str(nama_obat)
    nama_obat = nama_obat.lower().strip()
    nama_obat = nama_obat.replace("asetosal", "acetylsalicylic acid")
    nama_obat = nama_obat.replace("acetosal", "acetylsalicylic acid")
    nama_obat = re.sub(r"\s+", " ", nama_obat)
    return nama_obat


def ambil_zat_aktif(nama_obat):
    nama_normal = normalisasi_nama_obat(nama_obat)

    for zat_aktif in DAFTAR_ZAT_AKTIF_ACUAN:
        if nama_normal == zat_aktif or nama_normal.startswith(zat_aktif + " "):
            return zat_aktif

    # Fallback untuk nama yang belum terdapat pada daftar acuan.
    hasil = re.sub(
        r"\s+\d+(?:[.,]\d+)?(?:\s*[+/]\s*\d+(?:[.,]\d+)?)?\s*(?:mg|mcg|g|ml|iu(?:/ml)?)\b.*$",
        "",
        nama_normal
    )
    return hasil.strip()


def ambil_kekuatan_dalam_data(nama_obat, zat_aktif):
    nama_normal = normalisasi_nama_obat(nama_obat)

    if nama_normal.startswith(zat_aktif):
        kekuatan = nama_normal[len(zat_aktif):].strip()
    else:
        kekuatan = nama_normal

    return kekuatan if kekuatan else "-"


def buat_informasi_kewaspadaan(daftar_obat_unik):
    kekuatan_dalam_data = {}

    for nama_obat in daftar_obat_unik:
        zat_aktif = ambil_zat_aktif(nama_obat)
        kekuatan = ambil_kekuatan_dalam_data(nama_obat, zat_aktif)

        kekuatan_dalam_data.setdefault(zat_aktif, set()).add(kekuatan)

    zat_aktif_variasi_kekuatan = sorted(
        zat_aktif
        for zat_aktif in kekuatan_dalam_data
        if zat_aktif in VARIASI_KEKUATAN_ACUAN
    )

    zat_aktif_nama_mirip = sorted(
        zat_aktif
        for zat_aktif in kekuatan_dalam_data
        if zat_aktif in PASANGAN_MIRIP_MAP
    )

    seluruh_zat_aktif = sorted(
        set(zat_aktif_variasi_kekuatan) | set(zat_aktif_nama_mirip)
    )

    baris_detail = []

    for zat_aktif in seluruh_zat_aktif:
        kategori = []

        if zat_aktif in VARIASI_KEKUATAN_ACUAN:
            kategori.append("Variasi kekuatan")

        if zat_aktif in PASANGAN_MIRIP_MAP:
            kategori.append("Zat aktif mirip")

        daftar_kekuatan = sorted(kekuatan_dalam_data.get(zat_aktif, {"-"}))
        pasangan_mirip = sorted(PASANGAN_MIRIP_MAP.get(zat_aktif, set()))

        baris_detail.append({
            "Zat Aktif dalam Data": zat_aktif,
            "Kategori Informasi": "; ".join(kategori),
            "Kekuatan dalam Data": "; ".join(daftar_kekuatan),
            "Variasi Kekuatan Acuan": VARIASI_KEKUATAN_ACUAN.get(zat_aktif, "-"),
            "Pasangan Zat Aktif Mirip": "; ".join(pasangan_mirip) if pasangan_mirip else "-"
        })

    kolom_detail = [
        "Zat Aktif dalam Data",
        "Kategori Informasi",
        "Kekuatan dalam Data",
        "Variasi Kekuatan Acuan",
        "Pasangan Zat Aktif Mirip"
    ]

    informasi_df = pd.DataFrame(baris_detail, columns=kolom_detail)

    return informasi_df, zat_aktif_variasi_kekuatan, zat_aktif_nama_mirip


def buat_file_excel(
    rules_tampil,
    df_bersih,
    transaksi,
    daftar_obat_unik,
    informasi_kewaspadaan=None
):
    output = BytesIO()

    daftar_obat_unik_df = pd.DataFrame({
        "Nama Obat": daftar_obat_unik
    })

    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        rules_tampil.to_excel(writer, index=False, sheet_name="Aturan Asosiasi")
        df_bersih.to_excel(writer, index=False, sheet_name="Data Bersih")
        transaksi.to_excel(writer, index=False, sheet_name="Data Transaksi")
        daftar_obat_unik_df.to_excel(writer, index=False, sheet_name="Daftar Obat Unik")

        if informasi_kewaspadaan is not None:
            informasi_kewaspadaan.to_excel(
                writer,
                index=False,
                sheet_name="Informasi Kewaspadaan"
            )

    output.seek(0)
    return output


# NETWORK GRAPH #


def ambil_warna_consequent(rules):
    warna_default = {
        "metformin 500 mg": "#8c564b",
        "glimepiride 2 mg": "#7f7f7f",
        "candesartan 8 mg": "#9467bd",
        "acetylsalicylic acid 80 mg": "#1f77b4",
        "amlodipine 10 mg": "#d62728",
        "bisoprolol 2,5 mg": "#ff7f0e",
        "candesartan 16 mg": "#2ca02c"
    }

    warna_cadangan = [
        "#17becf",
        "#e377c2",
        "#bcbd22",
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#7f7f7f"
    ]

    daftar_consequent = []

    for cons in rules["consequents"]:
        for item in list(cons):
            if item not in daftar_consequent:
                daftar_consequent.append(item)

    warna_map = {}
    index_cadangan = 0

    for obat in daftar_consequent:
        if obat in warna_default:
            warna_map[obat] = warna_default[obat]
        else:
            warna_map[obat] = warna_cadangan[index_cadangan % len(warna_cadangan)]
            index_cadangan += 1

    return warna_map


def tampilkan_keterangan_network(rules):
    warna_map = ambil_warna_consequent(rules)

    st.markdown("**Keterangan Network Graph**")

    st.markdown(
        """
**Simbol:**
- Lingkaran biru menunjukkan obat pemicu (*antecedent*).
- Kotak oranye menunjukkan kode aturan asosiasi, misalnya R1, R2, dan seterusnya.
- Lingkaran abu-abu menunjukkan obat yang ikut muncul (*consequent*).
- Garis putus-putus menunjukkan arah dari obat dalam resep menuju kode aturan.
- Garis lurus menunjukkan arah dari kode aturan menuju obat yang ikut muncul.
- Garis tepi oranye menunjukkan zat aktif dengan variasi kekuatan sediaan.
- Garis tepi merah menunjukkan zat aktif yang memiliki pasangan nama mirip berdasarkan daftar ISMP.
- Garis tepi oranye dan merah menunjukkan zat aktif yang termasuk dalam kedua informasi tersebut.

**Cara baca:**
- Network graph dibaca dari garis putus-putus menuju ke kotak aturan, kemudian dari kotak aturan mengikuti garis lurus menuju ke obat tujuan.
- Contoh: garis putus-putus berasal dari obat A → R1 → garis lurus menuju ke obat B, berarti apabila dokter meresepkan obat A, maka obat B cenderung ikut diresepkan.
        """
    )

    st.markdown("**Warna garis berdasarkan consequent:**")

    item_warna = []
    for obat, warna in warna_map.items():
        item_warna.append(
            f"""
            <div style="display:flex; align-items:center; gap:7px; white-space:nowrap;">
                <span style="width:28px; height:0; border-top:3px solid {warna}; display:inline-block;"></span>
                <span>{obat}</span>
            </div>
            """
        )

    st.markdown(
        f"""
        <div style="display:flex; flex-wrap:wrap; column-gap:20px; row-gap:8px; margin-top:4px;">
            {''.join(item_warna)}
        </div>
        """,
        unsafe_allow_html=True
    )


def ambil_contoh_baca_network(rules2, jumlah=3):
    rules2 = rules2.copy()

    rules2["ant_list"] = rules2["antecedents"].apply(
        lambda x: list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]
    )

    rules2["con_list"] = rules2["consequents"].apply(
        lambda x: list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]
    )

    rules2["con_str"] = rules2["con_list"].apply(lambda x: x[0] if len(x) > 0 else "")

    right_order = [
        "glimepiride 2 mg",
        "metformin 500 mg",
        "candesartan 8 mg",
        "acetylsalicylic acid 80 mg",
        "amlodipine 10 mg",
        "bisoprolol 2,5 mg",
        "candesartan 16 mg"
    ]

    drug_order = [
        "glimepiride 1 mg",
        "glimepiride 2 mg",
        "furosemide 40 mg",
        "nifedipine 30 mg",
        "clopidogrel 75 mg",
        "bisoprolol 5 mg",
        "metformin 500 mg",
        "candesartan 8 mg",
        "acetylsalicylic acid 80 mg",
        "amlodipine 10 mg",
        "bisoprolol 2,5 mg",
        "candesartan 16 mg"
    ]

    all_ants = sorted(set(a for ants in rules2["ant_list"] for a in ants))
    all_cons = sorted(set(c for cons in rules2["con_list"] for c in cons))
    all_drugs = sorted(set(all_ants + all_cons))

    for d in all_drugs:
        if d not in drug_order:
            drug_order.append(d)

    for d in all_cons:
        if d not in right_order:
            right_order.append(d)

    drug_rank = {drug: i for i, drug in enumerate(drug_order)}
    right_rank = {drug: i for i, drug in enumerate(right_order)}

    def avg_ant_rank(ants):
        ranks = [drug_rank.get(a, 999) for a in ants]
        return np.mean(ranks) if len(ranks) > 0 else 999

    rules2["avg_ant_rank"] = rules2["ant_list"].apply(avg_ant_rank)
    rules2["con_rank"] = rules2["con_str"].apply(lambda x: right_rank.get(x, 999))

    top_rules = rules2.sort_values(
        by=["con_rank", "avg_ant_rank", "confidence", "lift"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)

    hasil = []

    for i, row in top_rules.head(jumlah).iterrows():
        rule_id = f"R{i + 1}"
        antecedent = ", ".join(row["ant_list"])
        consequent = ", ".join(row["con_list"])

        hasil.append({
            "rule_id": rule_id,
            "antecedent": antecedent,
            "consequent": consequent
        })

    return hasil

def buat_network_graph(rules2):
    rules2 = rules2.copy()

    rules2["ant_list"] = rules2["antecedents"].apply(
        lambda x: list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]
    )

    rules2["con_list"] = rules2["consequents"].apply(
        lambda x: list(x) if isinstance(x, (list, tuple, set, frozenset)) else [x]
    )

    rules2["con_str"] = rules2["con_list"].apply(lambda x: x[0] if len(x) > 0 else "")

    right_order = [
        "glimepiride 2 mg",
        "metformin 500 mg",
        "candesartan 8 mg",
        "acetylsalicylic acid 80 mg",
        "amlodipine 10 mg",
        "bisoprolol 2,5 mg",
        "candesartan 16 mg"
    ]

    drug_order = [
        "glimepiride 1 mg",
        "glimepiride 2 mg",
        "furosemide 40 mg",
        "nifedipine 30 mg",
        "clopidogrel 75 mg",
        "bisoprolol 5 mg",
        "metformin 500 mg",
        "candesartan 8 mg",
        "acetylsalicylic acid 80 mg",
        "amlodipine 10 mg",
        "bisoprolol 2,5 mg",
        "candesartan 16 mg"
    ]

    all_ants = sorted(set(a for ants in rules2["ant_list"] for a in ants))
    all_cons = sorted(set(c for cons in rules2["con_list"] for c in cons))
    all_drugs = sorted(set(all_ants + all_cons))

    for d in all_drugs:
        if d not in drug_order:
            drug_order.append(d)

    for d in all_cons:
        if d not in right_order:
            right_order.append(d)

    drug_rank = {drug: i for i, drug in enumerate(drug_order)}
    right_rank = {drug: i for i, drug in enumerate(right_order)}

    def avg_ant_rank(ants):
        ranks = [drug_rank.get(a, 999) for a in ants]
        return np.mean(ranks) if len(ranks) > 0 else 999

    rules2["avg_ant_rank"] = rules2["ant_list"].apply(avg_ant_rank)
    rules2["con_rank"] = rules2["con_str"].apply(lambda x: right_rank.get(x, 999))

    top_rules = rules2.sort_values(
        by=["con_rank", "avg_ant_rank", "confidence", "lift"],
        ascending=[True, True, False, False]
    ).reset_index(drop=True)

    top_rules["rule_id"] = ["R" + str(i + 1) for i in range(len(top_rules))]

    right_drugs = set(c for cons in top_rules["con_list"] for c in cons)
    all_ant_drugs = set(a for ants in top_rules["ant_list"] for a in ants)
    left_drugs = all_ant_drugs - right_drugs

    right_drugs_sorted = [d for d in right_order if d in right_drugs]
    left_drugs_sorted = [d for d in drug_order if d in left_drugs]

    G = nx.DiGraph()

    consequent_colors = consequent_colors = ambil_warna_consequent(rules2)

    for drug in left_drugs_sorted:
        G.add_node(drug, node_type="drug_left")

    for drug in right_drugs_sorted:
        G.add_node(drug, node_type="drug_right")

    for _, row in top_rules.iterrows():
        rule_node = row["rule_id"]
        ants = row["ant_list"]
        cons = row["con_list"]
        con_main = row["con_str"]
        conf = row["confidence"]
        color = consequent_colors.get(con_main, "black")

        G.add_node(
            rule_node,
            node_type="rule",
            confidence=conf,
            consequent=con_main,
            color=color
        )

        for ant in ants:
            G.add_edge(
                ant, rule_node,
                edge_type="ant",
                rule=rule_node,
                color=color
            )

        for con in cons:
            G.add_edge(
                rule_node, con,
                edge_type="cons",
                rule=rule_node,
                color=color
            )

    pos = {}

    rule_gap = 1.25
    group_gap = 0.70

    current_y = 0
    rule_y_map = {}
    group_centers = {}

    for con in right_drugs_sorted:
        subset = top_rules[top_rules["con_str"] == con]

        if len(subset) == 0:
            continue

        start_y = current_y

        for _, row in subset.iterrows():
            rid = row["rule_id"]
            pos[rid] = (0, -current_y)
            rule_y_map[rid] = -current_y
            current_y += rule_gap

        end_y = current_y - rule_gap
        group_centers[con] = -((start_y + end_y) / 2)

        current_y += group_gap

    for drug in right_drugs_sorted:
        if drug in group_centers:
            pos[drug] = (3.2, group_centers[drug])
        else:
            pos[drug] = (3.2, -current_y)
            current_y += 2.0

    left_anchor = {}

    for drug in left_drugs_sorted:
        connected_rules = []

        for _, row in top_rules.iterrows():
            if drug in row["ant_list"]:
                connected_rules.append(row["rule_id"])

        if connected_rules:
            left_anchor[drug] = np.mean([rule_y_map[r] for r in connected_rules])
        else:
            left_anchor[drug] = 0

    left_drugs_sorted = sorted(left_drugs_sorted, key=lambda d: left_anchor[d], reverse=True)

    min_left_gap = 1.85
    last_y = None

    for drug in left_drugs_sorted:
        y = left_anchor[drug]

        if last_y is not None and y > last_y - min_left_gap:
            y = last_y - min_left_gap

        pos[drug] = (-3.2, y)
        last_y = y

    def wrap_drug_label(label):
        label = str(label)
        parts = label.rsplit(" ", 2)

        if len(parts) == 3 and parts[-1] == "mg":
            return parts[0] + "\n" + parts[1] + " " + parts[2]
        else:
            return label

    labels = {}

    for node, data in G.nodes(data=True):
        if data.get("node_type") in ["drug_left", "drug_right"]:
            labels[node] = wrap_drug_label(node)
        else:
            labels[node] = node

    all_drug_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") in ["drug_left", "drug_right"]
    ]

    strength_warning_nodes = [
        n for n in all_drug_nodes
        if ambil_zat_aktif(n) in VARIASI_KEKUATAN_ACUAN
    ]

    similar_name_warning_nodes = [
        n for n in all_drug_nodes
        if ambil_zat_aktif(n) in PASANGAN_MIRIP_MAP
    ]

    both_warning_nodes = sorted(
        set(strength_warning_nodes) & set(similar_name_warning_nodes)
    )
    strength_only_nodes = sorted(
        set(strength_warning_nodes) - set(both_warning_nodes)
    )
    similar_only_nodes = sorted(
        set(similar_name_warning_nodes) - set(both_warning_nodes)
    )

    node_size_map = {}
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "drug_left":
            node_size_map[node] = 2400
        elif data.get("node_type") == "drug_right":
            node_size_map[node] = 2600

    fig, ax = plt.subplots(figsize=(13, 18))

    left_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "drug_left"
    ]

    right_nodes = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "drug_right"
    ]

    rule_nodes_graph = [
        n for n, d in G.nodes(data=True)
        if d.get("node_type") == "rule"
    ]

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=left_nodes,
        node_color="lightblue",
        node_size=2400,
        alpha=0.95,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=right_nodes,
        node_color="lightgray",
        node_size=2600,
        alpha=0.95,
        ax=ax
    )

    nx.draw_networkx_nodes(
        G, pos,
        nodelist=rule_nodes_graph,
        node_color="orange",
        node_shape="s",
        node_size=900,
        alpha=0.95,
        ax=ax
    )

    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
        ax=ax
    )

    for rule in rule_nodes_graph:
        color = G.nodes[rule].get("color", "black")

        ant_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "ant" and d.get("rule") == rule
        ]

        cons_edges = [
            (u, v) for u, v, d in G.edges(data=True)
            if d.get("edge_type") == "cons" and d.get("rule") == rule
        ]

        nx.draw_networkx_edges(
            G, pos,
            edgelist=ant_edges,
            edge_color=[color],
            arrowstyle="-|>",
            arrowsize=13,
            width=1.3,
            alpha=0.70,
            style="dashed",
            connectionstyle="arc3,rad=0.0",
            min_source_margin=35,
            min_target_margin=30,
            ax=ax
        )

        nx.draw_networkx_edges(
            G, pos,
            edgelist=cons_edges,
            edge_color=[color],
            arrowstyle="-|>",
            arrowsize=14,
            width=2.3,
            alpha=0.90,
            connectionstyle="arc3,rad=0.0",
            min_source_margin=25,
            min_target_margin=35,
            ax=ax
        )

    # Garis tepi informasi obat digambar setelah edge agar tetap terlihat jelas.
    strength_border_color = "#FF9D00"
    similar_border_color = "#C00000"
    ring_width = 7
    separator_width = 2

    def enlarged_node_size(base_size, radius_addition):
        return (np.sqrt(base_size) + radius_addition) ** 2

    def original_node_color(node):
        if G.nodes[node].get("node_type") == "drug_left":
            return "lightblue"
        return "lightgray"

    if strength_only_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=strength_only_nodes,
            node_color=strength_border_color,
            edgecolors="none",
            node_size=[
                enlarged_node_size(node_size_map[n], ring_width)
                for n in strength_only_nodes
            ],
            alpha=1,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=strength_only_nodes,
            node_color=[original_node_color(n) for n in strength_only_nodes],
            edgecolors="none",
            node_size=[node_size_map[n] for n in strength_only_nodes],
            alpha=0.95,
            ax=ax
        )

    if similar_only_nodes:
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=similar_only_nodes,
            node_color=similar_border_color,
            edgecolors="none",
            node_size=[
                enlarged_node_size(node_size_map[n], ring_width)
                for n in similar_only_nodes
            ],
            alpha=1,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=similar_only_nodes,
            node_color=[original_node_color(n) for n in similar_only_nodes],
            edgecolors="none",
            node_size=[node_size_map[n] for n in similar_only_nodes],
            alpha=0.95,
            ax=ax
        )

    if both_warning_nodes:
        # Lapisan terluar merah dan lapisan dalam oranye dibuat sama tebal.
        nx.draw_networkx_nodes(
            G, pos,
            nodelist=both_warning_nodes,
            node_color=similar_border_color,
            edgecolors="none",
            node_size=[
                enlarged_node_size(
                    node_size_map[n],
                    (2 * ring_width) + separator_width
                )
                for n in both_warning_nodes
            ],
            alpha=1,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=both_warning_nodes,
            node_color="white",
            edgecolors="none",
            node_size=[
                enlarged_node_size(
                    node_size_map[n],
                    ring_width + separator_width
                )
                for n in both_warning_nodes
            ],
            alpha=1,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=both_warning_nodes,
            node_color=strength_border_color,
            edgecolors="none",
            node_size=[
                enlarged_node_size(node_size_map[n], ring_width)
                for n in both_warning_nodes
            ],
            alpha=1,
            ax=ax
        )

        nx.draw_networkx_nodes(
            G, pos,
            nodelist=both_warning_nodes,
            node_color=[original_node_color(n) for n in both_warning_nodes],
            edgecolors="none",
            node_size=[node_size_map[n] for n in both_warning_nodes],
            alpha=0.95,
            ax=ax
        )

    # Gambar ulang label agar tidak tertutup oleh garis tepi informasi obat.
    nx.draw_networkx_labels(
        G, pos,
        labels=labels,
        font_size=10,
        font_weight="bold",
        ax=ax
    )

    ax.axis("off")

    if len(pos) > 0:
        y_vals = [p[1] for p in pos.values()]

        # Tambahkan ruang di atas dan bawah agar node serta garis tepi tidak terpotong
        padding_y = 1.8

        ax.set_ylim(
            min(y_vals) - padding_y,
            max(y_vals) + padding_y
        )

        ax.set_xlim(-4.4, 4.4)

    plt.tight_layout(pad=2.0)

    return fig


def tampilkan_fig_dalam_card(fig):
    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=160, facecolor="white")
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode("utf-8")

    st.markdown(
        f"""
        <div class="graph-card">
            <img src="data:image/png;base64,{img_base64}" alt="Network Graph Aturan Asosiasi">
        </div>
        """,
        unsafe_allow_html=True
    )


# TAMPILAN STREAMLIT #


st.markdown(
    """
    <style>
    /* ====== PANEL UTAMA ====== */
    .st-key-panel_ringkasan,
    .st-key-panel_model,
    .st-key-panel_model_empty,
    .st-key-panel_visualisasi {
        border: 1px solid rgba(120, 120, 120, 0.28) !important;
        border-radius: 16px !important;
        padding: 18px 20px 20px 20px !important;
        background-color: #f5f6f8 !important;
        margin-top: 14px !important;
        margin-bottom: 16px !important;
    }

    /* ====== PANEL INFORMASI KEWASPADAAN OBAT ====== */
    .st-key-panel_kewaspadaan {
        border: 1px solid rgba(120, 120, 120, 0.28) !important;
        border-radius: 16px !important;
        padding: 18px 20px 20px 20px !important;
        background-color: #f5f6f8 !important;
        margin-top: 14px !important;
        margin-bottom: 16px !important;
    }

    .st-key-panel_kewaspadaan > div {
        background-color: #f5f6f8 !important;
        border-radius: 16px !important;
    }

    /* isi dalam panel */
    .st-key-panel_ringkasan > div,
    .st-key-panel_model > div,
    .st-key-panel_model_empty > div,
    .st-key-panel_visualisasi > div {
        background-color: #f5f6f8 !important;
        border-radius: 16px !important;
    }

    .section-title {
        font-size: 23px;
        font-weight: 850;
        color: var(--text-color) !important;
        margin-bottom: 18px;
        text-transform: uppercase;
        letter-spacing: 0.3px;
        line-height: 1.25;
        padding: 11px 18px;
        border-left: none;
        border-radius: 10px;
        background: linear-gradient(
            90deg,
            rgba(46, 157, 87, 0.10) 0%,
            rgba(46, 157, 87, 0.06) 55%,
            rgba(46, 157, 87, 0.03) 100%
        );
    }

    .subsection-title {
        font-size: 18px;
        font-weight: 750;
        color: var(--text-color) !important;
        margin-top: 10px;
        margin-bottom: 10px;
        line-height: 1.35;
        padding: 0;
        border-left: none;
        border-radius: 0;
        background-color: transparent;
    }

    /* ====== CARD RINGKASAN DATA ====== */
    .metric-card {
        background-color: var(--background-color);
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-left: 4px solid #2e9d57;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: none;
        height: 112px;
        display: flex;
        flex-direction: column;
        justify-content: center;
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 13px;
        color: var(--text-color);
        opacity: 0.70;
        margin-bottom: 8px;
        font-weight: 650;
        min-height: 28px;
        display: flex;
        align-items: center;
        line-height: 1.35;
    }

    .metric-value {
        font-size: 29px;
        color: var(--text-color);
        font-weight: 800;
        line-height: 1.1;
    }

    /* ====== EXPANDER DAN TABEL ====== */
    div[data-testid="stExpander"] {
        margin-top: 8px !important;
        margin-bottom: 8px !important;
    }

    div[data-testid="stExpander"] details {
        background-color: var(--background-color) !important;
        border: 1px solid rgba(120, 120, 120, 0.18) !important;
        border-left: 4px solid #2e9d57 !important;
        border-radius: 12px !important;
        box-shadow: none !important;
    }

    div[data-testid="stExpander"] summary {
        font-size: 13px !important;
        font-weight: 650 !important;
        color: var(--text-color) !important;
        opacity: 0.78 !important;
        padding: 6px 8px !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stExpander"] summary p {
        font-weight: 800 !important;
    }

    div[data-testid="stExpander"] summary:hover {
        opacity: 1 !important;
        color: var(--text-color) !important;
    }

    /* ====== CARD INTERPRETASI DAN CARA BACA ====== */
    .interpretasi-card,
    .baca-network-card {
        background-color: var(--background-color);
        color: var(--text-color);
        border: 1px solid rgba(120, 120, 120, 0.18);
        border-left: 4px solid #2e9d57;
        border-radius: 12px;
        padding: 20px 24px;
        margin-top: 6px;
        margin-bottom: 12px;
        box-shadow: none;
        transition: all 0.25s ease;
    }

    .interpretasi-card:hover,
    .baca-network-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.10);
        border-color: rgba(120, 120, 120, 0.24);
    }

    .interpretasi-title,
    .baca-network-title {
        font-size: 16px;
        font-weight: 800;
        color: var(--text-color);
        margin-bottom: 14px;
        line-height: 1.4;
        max-width: 980px;
    }

    .interpretasi-card ul,
    .baca-network-card ul {
        margin-top: 8px;
        margin-bottom: 0px;
        padding-left: 26px;
        max-width: 980px;
    }

    .interpretasi-card li,
    .baca-network-card li {
        margin-bottom: 8px;
        line-height: 1.55;
        font-size: 14.5px;
        color: var(--text-color);
    }

    .interpretasi-card li:last-child,
    .baca-network-card li:last-child {
        margin-bottom: 0px;
    }

    .interpretasi-card b,
    .baca-network-card b {
        color: var(--text-color);
        font-weight: 800;
    }

    /* ====== GRAPH ====== */
    .graph-card {
        background-color: var(--background-color);
        border: 1px solid rgba(120, 120, 120, 0.24);
        border-radius: 16px;
        padding: 10px 12px;
        margin-top: 8px;
        margin-bottom: 18px;
        box-shadow: none;
        overflow-x: auto;
    }

    .graph-card img {
        width: 100%;
        max-width: 980px;
        display: block;
        margin-left: auto;
        margin-right: auto;
        border-radius: 10px;
    }

    /* ====== TAB DETAIL DATA ====== */
    button[data-baseweb="tab"] {
        color: var(--text-color) !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] {
        color: #1f77d0 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #1f77d0 !important;
        font-weight: 700 !important;
    }

    div[data-baseweb="tab-highlight"] {
        background-color: #1f77d0 !important;
    }

    /* ====== TOMBOL DOWNLOAD ====== */
    div.stDownloadButton {
        margin-top: 2px !important;
        margin-bottom: 4px !important;
    }

    div.stDownloadButton > button {
        background-color: #1f77d0 !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.58em 1.08em !important;
        font-weight: 600 !important;
        font-size: 14px !important;
    }

    div.stDownloadButton > button:hover {
        background-color: #155fa8 !important;
        color: #ffffff !important;
        border: none !important;
    }

    div.stDownloadButton > button p,
    div.stDownloadButton > button span,
    div.stDownloadButton > button div {
        color: #ffffff !important;
        font-weight: 600 !important;
    }


    /* ====== FIX DARK MODE: PANEL TETAP TERBACA ====== */
    .st-key-panel_kewaspadaan,
    .st-key-panel_kewaspadaan * {
        color: #111827 !important;
    }

    .st-key-panel_kewaspadaan,
    .st-key-panel_kewaspadaan > div {
        background-color: #f5f6f8 !important;
    }

    .st-key-panel_ringkasan,
    .st-key-panel_model,
    .st-key-panel_model_empty,
    .st-key-panel_visualisasi,
    .st-key-panel_ringkasan *,
    .st-key-panel_model *,
    .st-key-panel_model_empty *,
    .st-key-panel_visualisasi * {
        color: #111827 !important;
    }

    .st-key-panel_ringkasan,
    .st-key-panel_model,
    .st-key-panel_model_empty,
    .st-key-panel_visualisasi,
    .st-key-panel_ringkasan > div,
    .st-key-panel_model > div,
    .st-key-panel_model_empty > div,
    .st-key-panel_visualisasi > div {
        background-color: #f5f6f8 !important;
    }

    .section-title {
        color: #111827 !important;
        background: linear-gradient(
            90deg,
            rgba(46, 157, 87, 0.16) 0%,
            rgba(46, 157, 87, 0.10) 55%,
            rgba(46, 157, 87, 0.06) 100%
        ) !important;
    }

    .subsection-title,
    .metric-label,
    .metric-value,
    .interpretasi-title,
    .baca-network-title,
    .interpretasi-card li,
    .baca-network-card li,
    .interpretasi-card b,
    .baca-network-card b {
        color: #111827 !important;
    }

    .metric-card,
    .interpretasi-card,
    .baca-network-card,
    .graph-card,
    div[data-testid="stExpander"] details {
        background-color: #ffffff !important;
        color: #111827 !important;
    }

    div[data-testid="stExpander"] summary,
    div[data-testid="stExpander"] summary p,
    div[data-testid="stExpander"] svg {
        color: #111827 !important;
        fill: #111827 !important;
    }

    button[data-baseweb="tab"],
    button[data-baseweb="tab"] p {
        color: #111827 !important;
    }

    button[data-baseweb="tab"][aria-selected="true"],
    button[data-baseweb="tab"][aria-selected="true"] p {
        color: #1f77d0 !important;
    }

    /* ====== WARNA TAB AKTIF: PAKSA BIRU ====== */
    html body [data-testid="stTabs"] [data-baseweb="tab-highlight"] {
        display: none !important;
    }

    html body [data-testid="stTabs"] button[role="tab"] {
        box-shadow: none !important;
        border-bottom: none !important;
    }

    html body [data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #1f77d0 !important;
        box-shadow: inset 0 -2px 0 #1f77d0 !important;
    }

    html body [data-testid="stTabs"] button[role="tab"][aria-selected="true"] p {
        color: #1f77d0 !important;
        font-weight: 700 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


uploaded_file = st.file_uploader(
    "Upload file data resep",
    type=["xlsx", "xls", "csv"],
    help="File harus berisi kolom No. Fraktur dan Produk."
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df_upload = pd.read_csv(uploaded_file)
    else:
        df_upload = pd.read_excel(uploaded_file)

    st.success("File berhasil dibaca.")

    try:
        df_detail, df_bersih, transaksi = proses_preprocessing(df_upload)

        daftar_obat_unik = hitung_obat_unik_dari_transaksi(transaksi)
        jumlah_obat_unik_transaksi = len(daftar_obat_unik)
        total_resep = transaksi["ID Resep"].nunique()
        total_resep_teks = format_bilangan(total_resep)

        informasi_kewaspadaan, zat_aktif_variasi_kekuatan, zat_aktif_nama_mirip = (
            buat_informasi_kewaspadaan(daftar_obat_unik)
        )

        tabel_variasi_kekuatan_acuan = buat_tabel_variasi_kekuatan_acuan()
        tabel_zat_aktif_mirip_acuan = buat_tabel_zat_aktif_mirip_acuan()

        # RINGKASAN DATA
        with st.container(border=True, key="panel_ringkasan"):
            st.markdown('<div class="section-title">Ringkasan Data</div>', unsafe_allow_html=True)

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Jumlah Resep</div>
                        <div class="metric-value">{format_bilangan(total_resep)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Jumlah Variasi Obat</div>
                        <div class="metric-value">{format_bilangan(jumlah_obat_unik_transaksi)}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)

            with st.expander("Lihat Detail Data"):
                tab_obat, tab_mentah, tab_bersih, tab_transaksi = st.tabs([
                    "Daftar Variasi Obat",
                    "Data Mentah",
                    "Data Bersih",
                    "Data Transaksi"
                ])

                with tab_obat:
                    daftar_obat_unik_tampil = pd.DataFrame({"Nama Obat": daftar_obat_unik})
                    daftar_obat_unik_tampil.index = range(1, len(daftar_obat_unik_tampil) + 1)
                    st.dataframe(daftar_obat_unik_tampil, use_container_width=True)

                with tab_mentah:
                    df_upload_tampil = df_upload.head(100).copy()
                    df_upload_tampil.index = range(1, len(df_upload_tampil) + 1)
                    st.dataframe(df_upload_tampil, use_container_width=True)

                with tab_bersih:
                    df_bersih_tampil = df_bersih.head(100).copy()
                    df_bersih_tampil.index = range(1, len(df_bersih_tampil) + 1)
                    st.dataframe(df_bersih_tampil, use_container_width=True)

                with tab_transaksi:
                    transaksi_tampil = transaksi.head(100).copy()
                    transaksi_tampil.index = range(1, len(transaksi_tampil) + 1)
                    st.dataframe(transaksi_tampil, use_container_width=True)

        # INFORMASI KEWASPADAAN OBAT
        with st.container(border=True, key="panel_kewaspadaan"):
            st.markdown(
                '<div class="section-title">Informasi Kewaspadaan Obat</div>',
                unsafe_allow_html=True
            )

            col1, col2 = st.columns(2)

            with col1:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Zat Aktif dengan Variasi Kekuatan</div>
                        <div class="metric-value">{format_bilangan(len(tabel_variasi_kekuatan_acuan))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-label">Zat Aktif dengan Nama Mirip</div>
                        <div class="metric-value">{format_bilangan(len(tabel_zat_aktif_mirip_acuan))}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown(
                """
                Informasi ini memuat daftar zat aktif yang memiliki lebih dari satu kekuatan sediaan
                serta pasangan zat aktif dengan kemiripan nama berdasarkan daftar ISMP. Informasi tersebut
                dapat dijadikan sebagai perhatian tambahan untuk mendukung kehati-hatian dalam proses penginputan,
                pemilihan, pengambilan, penyiapan, dan penyerahan obat.
                """
            )

            with st.expander("Lihat Detail Informasi Obat"):
                tab_variasi, tab_mirip = st.tabs([
                    "Variasi Kekuatan",
                    "Zat Aktif Mirip"
                ])

                with tab_variasi:
                    informasi_variasi_tampil = tabel_variasi_kekuatan_acuan.copy()
                    informasi_variasi_tampil.index = range(
                        1,
                        len(informasi_variasi_tampil) + 1
                    )
                    st.dataframe(
                        informasi_variasi_tampil,
                        use_container_width=True
                    )

                with tab_mirip:
                    informasi_mirip_tampil = tabel_zat_aktif_mirip_acuan.copy()
                    informasi_mirip_tampil.index = range(
                        1,
                        len(informasi_mirip_tampil) + 1
                    )
                    st.dataframe(
                        informasi_mirip_tampil,
                        use_container_width=True
                    )


        frequent_itemsets, rules = proses_arm(
            transaksi,
            min_support=0.026,
            min_confidence=0.20
        )

        if rules.empty:
            with st.container(border=True, key="panel_model_empty"):
                st.markdown('<div class="section-title">Model Prediksi</div>', unsafe_allow_html=True)
                st.warning("Tidak ada aturan asosiasi yang terbentuk dengan parameter saat ini.")
        else:
            # MODEL PREDIKSI
            with st.container(border=True, key="panel_model"):
                st.markdown('<div class="section-title">Model Prediksi</div>', unsafe_allow_html=True)
                st.markdown('<div class="subsection-title">Hasil Aturan Asosiasi</div>', unsafe_allow_html=True)

                rules_tampil = format_rules_for_display(rules)
                rules_tampil.index = range(1, len(rules_tampil) + 1)
                st.dataframe(rules_tampil, use_container_width=True)

                excel_file = buat_file_excel(
                    rules_tampil=rules_tampil,
                    df_bersih=df_bersih,
                    transaksi=transaksi,
                    daftar_obat_unik=daftar_obat_unik,
                    informasi_kewaspadaan=informasi_kewaspadaan
                )

                st.download_button(
                    label="Download Hasil",
                    data=excel_file,
                    file_name="hasil_aturan_asosiasi_obat_bpjs_prb.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )

                st.markdown("<div style='height: 2px;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="subsection-title">Contoh Interpretasi 3 Aturan Asosiasi</div>', unsafe_allow_html=True)

                interpretasi = buat_interpretasi_rules(rules, total_resep=total_resep, jumlah=3)

                for item in interpretasi:
                    st.markdown(
                        f"""
                        <div class="interpretasi-card">
                            <div class="interpretasi-title">
                                Aturan {item["nomor"]}: IF {item["antecedent"]} THEN {item["consequent"]}
                            </div>
                            <ul>
                                <li>Nilai support {item["support"]} menunjukkan bahwa kombinasi tersebut muncul pada <b>{format_bilangan(item["jumlah_kombinasi"])} dari {total_resep_teks} resep</b>.</li>
                                <li>Nilai confidence {item["confidence"]} menunjukkan bahwa dari <b>{format_bilangan(item["jumlah_antecedent"])} resep</b> yang berisi <b>{item["antecedent"]}</b>, terdapat <b>{format_bilangan(item["jumlah_confidence"])} resep</b> yang juga berisi <b>{item["consequent"]}</b>.</li>
                                <li>Nilai lift {item["lift"]} menunjukkan bahwa {item["antecedent"]} dan {item["consequent"]} memiliki kecenderungan <b>muncul bersamaan dalam resep</b>.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

            # VISUALISASI
            with st.container(border=True, key="panel_visualisasi"):
                st.markdown('<div class="section-title">Visualisasi</div>', unsafe_allow_html=True)
                st.markdown('<div class="subsection-title">Network Graph Aturan Asosiasi</div>', unsafe_allow_html=True)

                fig = buat_network_graph(rules)
                tampilkan_fig_dalam_card(fig)

                st.markdown(
                    """
                    <div class="interpretasi-card">
                        <div class="interpretasi-title">
                            ⚠️ Informasi Kewaspadaan
                        </div>
                        <ul>
                            <li>Obat dengan <b>garis tepi oranye</b> memiliki <b>variasi kekuatan sediaan</b>, sehingga diperlukan kehati-hatian saat menginput, memilih, mengambil, menyiapkan, dan menyerahkan obat.</li>
                            <li>Obat dengan <b>garis tepi merah</b> memiliki <b>pasangan zat aktif dengan nama mirip</b>, sehingga diperlukan kehati-hatian saat membaca nama, memilih, mengambil, menyiapkan, dan menyerahkan obat.</li>
                            <li>Obat dengan <b>garis tepi oranye dan merah</b> memiliki kedua karakteristik tersebut, sehingga verifikasi nama zat aktif dan kekuatan sediaan perlu dilakukan dengan lebih teliti.</li>
                        </ul>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
                st.markdown('<div class="subsection-title">Contoh Cara Membaca Network Graph</div>', unsafe_allow_html=True)

                contoh_baca_network = ambil_contoh_baca_network(rules, jumlah=3)

                for item in contoh_baca_network:
                    st.markdown(
                        f"""
                        <div class="baca-network-card">
                            <div class="baca-network-title">
                                Aturan {item["rule_id"]}: IF {item["antecedent"]} THEN {item["consequent"]}
                            </div>
                            <ul>
                                <li>Pada {item["rule_id"]}, garis putus-putus berasal dari {item["antecedent"]} menuju {item["rule_id"]}, kemudian garis lurus dari {item["rule_id"]} mengarah ke {item["consequent"]}.</li>
                                <li>Artinya, apabila dokter meresepkan {item["antecedent"]}, maka {item["consequent"]} cenderung ikut diresepkan.</li>
                            </ul>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    except Exception as e:
        st.error("Terjadi error saat proses analisis.")
        st.exception(e)

else:
    st.info("Silakan upload file data resep terlebih dahulu.")
