"""Тесты загрузчика данных."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd

from src.data.loader import get_download_url, load_data


def test_get_download_url_returns_string():
    """Проверяет, что API возвращает строку."""
    with patch("src.data.loader.requests.get") as mock_get:
        mock_get.return_value.json.return_value = {
            "href": "http://example.com/file.xlsx"
        }
        mock_get.return_value.raise_for_status = MagicMock()
        url = get_download_url("https://disk.yandex.ru/i/abc")
        assert url == "http://example.com/file.xlsx"


def test_load_data_reads_existing_file(tmp_path: Path):
    """Проверяет, что load_data читает уже скачанный файл."""
    xlsx_path = tmp_path / "test.xlsx"
    pd.DataFrame({
        "Age": [25, 30],
        "RiskLevel": ["low risk", "high risk"],
    }).to_excel(xlsx_path, sheet_name="Sheet1", index=False)

    df = load_data(save_path=xlsx_path, sheet_name="Sheet1")

    assert df.shape == (2, 2)
    assert list(df.columns) == ["Age", "RiskLevel"]
    assert df["Age"].tolist() == [25, 30]