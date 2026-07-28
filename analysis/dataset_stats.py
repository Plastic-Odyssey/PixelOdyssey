from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from shapely.geometry import Polygon

# --- CONFIGURATION ---
CLASS_NAMES = {
    0: "bouteille PET",
    1: "Cordage",
    2: "Bouées",
    3: "Inconnu",
    4: "Fragments plastique rigide",
    5: "Bouées pieuvre",
    6: "Petit bidon",
    7: "Plastique souple (sac)",
    8: "Mousse expansée",
    9: "Sceau",
    10: "Polystyrene",
    11: "fliflops",
    12: "casques",
    13: "bouchons",
    14: "cagette",
    15: "Bouteille Plastique rigide",
    16: "Morceaux de bois",
    17: "Verre",
    18: "Tissu",
    19: "flotteurs",
}

# Dimensions des tuiles (en pixels)
TILE_WIDTH = 640
TILE_HEIGHT = 640

# --- RESOLUTION SPATIALE (GSD) ---
GSD_CM_PER_PX = 0.5  # 1 px = 0.5 cm en largeur/hauteur
PX2_TO_CM2 = GSD_CM_PER_PX**2  # 1 px² = 0.25 cm²
CM2_TO_M2 = 1 / 10000  # 1 m² = 10 000 cm²

# Dossier racine contenant labels/train et labels/val
LABELS_ROOT_DIR = Path(
    r"E:\PixelOdyssey\3. Processed dataset\sliced_dataset\labels"
)


def analyze_yolo_segmentation(
    labels_root: Path, tile_w: int, tile_h: int, p2c_ratio: float
):
    records = []
    splits = ["train", "val"]

    for split in splits:
        split_dir = labels_root / split
        if not split_dir.exists():
            print(f"⚠️ Dossier non trouvé : {split_dir}")
            continue

        txt_files = list(split_dir.glob("*.txt"))
        print(f"🔍 Analyse du split '{split}' ({len(txt_files)} fichiers)...")

        for txt_file in txt_files:
            with open(txt_file, "r") as f:
                lines = f.readlines()

            for line in lines:
                parts = line.strip().split()
                if len(parts) < 7:  # Min 3 points (6 coords) + class_id
                    continue

                class_id = int(parts[0])
                coords = np.array([float(x) for x in parts[1:]]).reshape(-1, 2)

                # Conversion des coordonnées normalisées [0,1] en pixels
                coords[:, 0] *= tile_w
                coords[:, 1] *= tile_h

                poly = Polygon(coords)

                if poly.is_valid and poly.area > 0:
                    area_px = poly.area
                    area_cm2 = area_px * p2c_ratio

                    records.append(
                        {
                            "split": split,
                            "file": txt_file.name,
                            "class_id": class_id,
                            "class_name": CLASS_NAMES.get(
                                class_id, f"Classe_{class_id}"
                            ),
                            "area_px": area_px,
                            "area_cm2": area_cm2,
                        }
                    )

    return pd.DataFrame(records)


# --- EXÉCUTION DE L'ANALYSE ---
df_items = analyze_yolo_segmentation(
    LABELS_ROOT_DIR, TILE_WIDTH, TILE_HEIGHT, PX2_TO_CM2
)

if df_items.empty:
    print("❌ Aucune donnée trouvée. Vérifiez le chemin d'accès.")
else:
    # --- AGRÉGATION GLOBALE ---
    stats = (
        df_items.groupby(["class_id", "class_name"])
        .agg(
            count=("area_cm2", "count"),
            total_area_cm2=("area_cm2", "sum"),
            median_area_cm2=("area_cm2", "median"),
            mean_area_cm2=("area_cm2", "mean"),
        )
        .reset_index()
    )

    total_count = stats["count"].sum()
    total_area_cm2 = stats["total_area_cm2"].sum()
    total_area_m2 = total_area_cm2 * CM2_TO_M2

    stats["pct_count"] = (stats["count"] / total_count) * 100
    stats["pct_area"] = (stats["total_area_cm2"] / total_area_cm2) * 100

    # Convertir total_area_cm2 en m² pour une lecture plus intuitive
    stats["total_area_m2"] = stats["total_area_cm2"] * CM2_TO_M2

    # Tri par nombre d'items
    stats = stats.sort_values(by="count", ascending=False)

    print("\n=== 📊 TABLEAU RÉCAPITULATIF DES SURFACES (EN CM² / M²) ===")
    print(f"Nombre total d'objets : {total_count}")
    print(
        f"Surface totale couverte : {total_area_cm2:.1f} cm² ({total_area_m2:.2f} m²)\n"
    )

    # Affichage personnalisé des colonnes clés
    print(
        stats[
            [
                "class_name",
                "count",
                "pct_count",
                "pct_area",
                "median_area_cm2",
                "mean_area_cm2",
                "total_area_m2",
            ]
        ].to_string(
            index=False,
            header=[
                "Classe",
                "Items",
                "% Items",
                "% Surf.",
                "Médiane (cm²)",
                "Moyenne (cm²)",
                "Total (m²)",
            ],
            formatters={
                "pct_count": "{:.1f}%".format,
                "pct_area": "{:.1f}%".format,
                "median_area_cm2": "{:.1f}".format,
                "mean_area_cm2": "{:.1f}".format,
                "total_area_m2": "{:.3f}".format,
            },
        )
    )

    # --- VISUALISATION GRAPHIQUE ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

    # 1. Nombre d'items (%)
    sns.barplot(
        data=stats,
        y="class_name",
        x="pct_count",
        ax=axes[0],
        palette="viridis",
        hue="class_name",
        legend=False,
    )
    axes[0].set_title(
        "1. Répartition par Nombre d'Items (%)", fontsize=13, fontweight="bold"
    )
    axes[0].set_xlabel("% du nombre total d'objets")
    axes[0].set_ylabel("")

    for p in axes[0].patches:
        w = p.get_width()
        if w > 0:
            axes[0].annotate(
                f"{w:.1f}%",
                (w, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
            )

    # 2. Surface totale en cm² / m² (%)
    sns.barplot(
        data=stats,
        y="class_name",
        x="pct_area",
        ax=axes[1],
        palette="magma",
        hue="class_name",
        legend=False,
    )
    axes[1].set_title(
        "2. Répartition par Surface Cumulée (%)", fontsize=13, fontweight="bold"
    )
    axes[1].set_xlabel("% de la surface totale couverte")
    axes[1].set_ylabel("")

    for p in axes[1].patches:
        w = p.get_width()
        if w > 0:
            axes[1].annotate(
                f"{w:.1f}%",
                (w, p.get_y() + p.get_height() / 2.0),
                ha="left",
                va="center",
                xytext=(5, 0),
                textcoords="offset points",
                fontsize=9,
            )

    axes[0].set_xlim(0, stats["pct_count"].max() * 1.15)
    axes[1].set_xlim(0, stats["pct_area"].max() * 1.15)

    plt.suptitle(
        f"Analyse Physique du Dataset (GSD = {GSD_CM_PER_PX} cm/px | Total: {total_area_m2:.2f} m² de plastique)",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    plt.tight_layout()
    plt.show()