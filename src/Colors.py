# Description: This file is just meant to store a list of distinguishable
# colors for use in plotting multiple data series in the same plot.

dist_colors = [
	"#F0A3FF", "#0075DC", "#993F00",
	"#4C005C", "#191919", "#005C31",
	"#2BCE48", "#FFCC99", "#808080",
	"#94FFB5", "#8F7C00", "#9DCC00",
	"#C20088", "#003380", "#FFA405",
	"#FFA8BB", "#426600", "#FF0010",
	"#5EF1F2", "#00998F", "#E0FF66",
	"#740AFF", "#990000", "#FF5005"
]

def torgb(h):
	return [int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)]