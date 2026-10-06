"""Загрузка данных с Яндекс.Диска."""

import urllib.parse
from pathlib import Path

import pandas as pd
import requests

from src.config import PUBLIC_LINK, SHEET_NAME, DATA_FILE


def get_download_url(public_link: str) -> str:
    """Получает прямую ссылку на скачивание через REST API Яндекс.Диска.

    Args:
        public_link: публичная ссылка на файл или папку.

    Returns:
        Прямая ссылка на скачивание файла.
    """
    base_url = "https://cloud-api.yandex.net/v1/disk/public/resources/download?"
    params = urllib.parse.urlencode(dict(public_key=public_link))
    response = requests.get(base_url + params, timeout=30)
    response.raise_for_status()
    return response.json()["href"]


def download_file(url: str, save_path: Path) -> Path:
    """Скачивает файл по прямой ссылке и сохраняет на диск.

    Args:
        url: прямая ссылка на файл.
        save_path: путь, куда сохранить.

    Returns:
        Путь к сохранённому файлу.
    """
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    save_path.write_bytes(response.content)
    return save_path


def load_data(
    public_link: str = PUBLIC_LINK,
    save_path: Path = DATA_FILE,
    sheet_name: str = SHEET_NAME,
) -> pd.DataFrame:
    """Загружает датасет Maternal Health Risk.

    Если файл уже скачан — читает с диска. Иначе скачивает с Яндекс.Диска.

    Args:
        public_link: публичная ссылка на Яндекс.Диск.
        save_path: путь для сохранения файла.
        sheet_name: имя листа в Excel.

    Returns:
        DataFrame с исходными данными.
    """
    if not save_path.exists():
        url = get_download_url(public_link)
        download_file(url, save_path)
    return pd.read_excel(save_path, sheet_name=sheet_name)
