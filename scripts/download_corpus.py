"""Descarga del corpus oficial AEAT y DGII a data/raw/.

Uso:
    python scripts/download_corpus.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import requests
from tqdm import tqdm


CORPUS = [
    {
        "name": "ManualRenta2024Tomo1_ES.pdf",
        "url": "https://sede.agenciatributaria.gob.es/static_files/Sede/Biblioteca/Manual/Practicos/IRPF/IRPF-2024/ManualRenta2024Tomo1_es_es.pdf",
        "pais": "ES",
        "descripcion": "Manual Renta 2024 — IRPF general (AEAT)",
    },
    {
        "name": "ManualRenta2024Tomo2_ES.pdf",
        "url": "https://sede.agenciatributaria.gob.es/static_files/Sede/Biblioteca/Manual/Practicos/IRPF/IRPF-2024-Deducciones-autonomicas/ManualRenta2024Tomo2_es_es.pdf",
        "pais": "ES",
        "descripcion": "Manual Renta 2024 — Deducciones autonómicas (AEAT)",
    },
    # NOTA: Las URLs de la DGII pueden cambiar. Si fallan, descargar manualmente
    # desde https://dgii.gov.do/legislacion/codigoTributario/ y colocarlos en data/raw/
    {
        "name": "CodigoTributario_Ley11-92_DO.pdf",
        "url": "https://dgii.gov.do/legislacion/codigoTributario/Documents/Titulo1.pdf",
        "pais": "DO",
        "descripcion": "Código Tributario Dominicano (Ley 11-92, Título I)",
    },
]


def download_file(url: str, dest: Path) -> bool:
    """Descarga un archivo mostrando barra de progreso."""
    try:
        resp = requests.get(url, stream=True, timeout=60, allow_redirects=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            desc=dest.name,
            total=total,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for chunk in resp.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main() -> int:
    raw_dir = Path("data/raw")
    raw_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Descarga del corpus oficial AEAT y DGII")
    print("=" * 70)

    successes = 0
    for doc in CORPUS:
        dest = raw_dir / doc["name"]
        if dest.exists():
            print(f"✓ Ya existe: {doc['name']}")
            successes += 1
            continue
        print(f"\n→ Descargando {doc['name']} ({doc['descripcion']})...")
        if download_file(doc["url"], dest):
            successes += 1
            print(f"  ✓ Guardado en {dest}")
        else:
            print(
                f"  ⚠ Si la descarga falla (URL cambiada), descargue manualmente "
                f"desde el portal oficial y guárdelo como {dest}"
            )

    print("\n" + "=" * 70)
    print(f"Resumen: {successes}/{len(CORPUS)} documentos disponibles en {raw_dir}")
    print("Siguiente paso: python -m app.core.ingest --source raw")
    print("=" * 70)
    return 0 if successes > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
