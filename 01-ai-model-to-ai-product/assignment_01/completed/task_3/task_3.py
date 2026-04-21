import os
from dotenv import load_dotenv
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(SCRIPT_DIR, ".env"))

def main():
    # Cost calculation
    input_price_per_token = 0.03 / 1_000_000   # $0.03 per 1M input tokens
    output_price_per_token = 0.09 / 1_000_000  # $0.09 per 1M output tokens

    df_eval = pd.read_excel(os.path.join(SCRIPT_DIR, "assignment_01.xlsx"))

    df_eval["Cost"] = (
        df_eval["input_tokens"] * input_price_per_token
        + df_eval["output_tokens"] * output_price_per_token
    )

    df_eval.to_excel(os.path.join(SCRIPT_DIR, "assignment_01.xlsx"), index=False)
    df_eval[["product_name", "input_tokens", "output_tokens", "Cost"]]

if __name__ == "__main__":
    main()