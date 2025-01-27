import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def read_forces_csv(
		csv_filename: str,
		header_row_count: int,
		tare: list[str] | None = None
) -> tuple[pd.DataFrame, dict[str, list]]:
	"""
	reads the csv_filename and extracts the header information alongside a forces dataframe
	:param csv_filename: forces csv file to read
	:param header_row_count: the size of the header
	:return: a tuple of header dictionary and forces dataframe
	:param tare: for every col name in the list, normalizes w.r.t the first entry in the column
	"""
	# Extract relevant information from the header:
	with open(csv_filename, 'r') as file:
		header_lines = file.readlines()[:header_row_count]
	header_dict = {}
	for line in header_lines:
		sp = line.split(':,')
		if len(sp) == 2:
			value_split = sp[1].strip('\n').split(',')
			header_dict[sp[0]] = value_split

	# skip the header and parse the forces to a df:
	f1_col_ind = 1  # TODO: is this always the case?
	f4_col_ind = f1_col_ind + 4
	forces_df = pd.read_csv(csv_filename, skiprows=header_row_count).iloc[:-1, f1_col_ind:f4_col_ind]
	forces_df.columns = ['F1', 'F2', 'F3', 'F4']
	forces_df = forces_df.applymap(lambda x: float(x.replace(' N', '')))  # remove ` N` from every cell
	if tare:
		forces_df[tare] -= forces_df[tare].iloc[0]
	return forces_df, header_dict

def plot_forces(
		df: pd.DataFrame,
		header: dict[str, list],
		convert_to_time: bool = True,
		line_width: float = 3.0,
		legend_font_size: int = 16  # Adjust the legend font size as needed
):
	sample_rate = int(header['Sampling rate'][0].split('/')[0])  # extracting int value frequency
	start_time = " ".join(header['Start (Date Time)'])

	if convert_to_time:
		df.index = np.arange(0, len(df)) / sample_rate

	mpl.use("TkAgg")
	ax = df.plot(
		title=f"Forces [{sample_rate} sample rate], [{start_time}]",
		xlabel="Time [sec]" if convert_to_time else f"Sample [#]",
		grid=True,
		linewidth=line_width
	)

	# Increase legend font size
	ax.legend(fontsize=legend_font_size)

	plt.show()


if __name__ == '__main__':
	df, head = read_forces_csv(
		r'G:\My Drive\Master\Lab\Thesis\Forces\experiments\23_10_2023\Forces[F=9.701_A=M_PIdiv4.743_K=0.491].csv', 21)
	plot_forces(df, head)
