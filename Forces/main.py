import os.path

import pandas as pd

from Forces import parse_forces
from loguru import logger


def get_forces(
		exp_date,
		parent_dirname: str,
		photos_sub_dirname: str,
		show_force_results: bool,
		header_row_count: int,
		tare: list[str]
):
	force_df_path = f"{parent_dirname}\\experiments\\{exp_date}\\results\\{photos_sub_dirname.split('Photos')[1]}\\forces.pkl"
	if os.path.exists(force_df_path):
		logger.info(f"Loading {force_df_path} from memory...")
		return pd.read_pickle(force_df_path)

	forces_csv = f"{parent_dirname}\\experiments\\{exp_date}\\forces\\{photos_sub_dirname.replace('Photos', 'Forces')}.csv"
	logger.info(f'Extracting Forces from {forces_csv}...')
	df, header = parse_forces.read_forces_csv(
		csv_filename=forces_csv,
		header_row_count=header_row_count,
		tare=tare
	)
	if show_force_results:
		parse_forces.plot_forces(df, header)
	df.to_pickle(force_df_path)
	return df
