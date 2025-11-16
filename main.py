from __future__ import annotations

from typing import Iterable

import pandas as pd

from scanner import IndicatorConfig, scan_markets


def _print_section(title: str, df: pd.DataFrame, max_rows: int = 10) -> None:
    if df.empty:
        print(f"- {title}: 해당되는 종목이 없습니다.")
        return
    print(f"\n[{title}] (상위 {min(max_rows, len(df))}개)")
    print(df.head(max_rows).to_string(index=False))


def _build_sections(result: pd.DataFrame) -> Iterable[tuple[str, pd.DataFrame]]:
    oversold = result[result["RSI 시그널"] == "과매도"].sort_values("RSI")
    overbought = result[result["RSI 시그널"] == "과매수"].sort_values("RSI", ascending=False)
    stochastic_oversold = result[result["스토캐스틱 시그널"] == "과매도"].sort_values("%K")
    lower_band = result[result["볼린저 상태"] == "하단 돌파 (과매도)"]
    upper_band = result[result["볼린저 상태"] == "상단 돌파 (과매수)"]

    return (
        ("RSI 과매도", oversold),
        ("RSI 과매수", overbought),
        ("스토캐스틱 과매도", stochastic_oversold),
        ("볼린저 하단 돌파", lower_band),
        ("볼린저 상단 돌파", upper_band),
    )


def main() -> None:
    config = IndicatorConfig()
    result = scan_markets(config=config)

    if result.empty:
        print("지표를 계산할 수 있는 종목 데이터를 가져오지 못했습니다.")
        return

    print("KOSPI/KOSDAQ 전체 종목 스캔 결과")
    print(f"총 {len(result)}개 종목 분석 완료")

    for title, df in _build_sections(result):
        _print_section(title, df)

    csv_path = "scan_result.csv"
    result.to_csv(csv_path, index=False)
    print(f"\n전체 결과는 {csv_path} 파일로 저장되었습니다.")


if __name__ == "__main__":
    main()
