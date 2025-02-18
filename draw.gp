# 设置输出图像格式（例如 PNG）
set terminal pngcairo size 600, 600
set output 'scatter_plot.png'

# 设置图表标题
set title "Scatter Plot with Categories"

# 设置x、y轴标签
set xlabel "X Axis"
set ylabel "Y Axis"


# 设置坐标轴范围（可以根据你的数据进行调整）
set xrange [-4:4]
set yrange [-4:4]

# 根据第三列 (category) 的值区分颜色或样式
# 这里使用不同的符号来代表不同的类别
plot './outPutForDraw.txt' using 1:2:3 with points pt 5 ps 0.4 lc variable notitle
