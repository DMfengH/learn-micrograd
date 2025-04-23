# 设置输出图像格式（例如 PNG）
set terminal pngcairo size 1200, 600

# 设置输出文件
set output 'scatter_plot.png'

# 启用多图模式
set multiplot layout 1,2 title "Two Plots"

# 设置图表标题；设置x、y轴标签；设置坐标轴范围（可以根据你的数据进行调整）
set title "Scatter Plot with Categories"
set xlabel "X Axis"
set ylabel "Y Axis"
set xrange [-4:4]
set yrange [-4:4]

# 根据第三列 (category) 的值区分颜色或样式
# 这里使用不同的符号来代表不同的类别
plot './outPutForDraw.txt' using 1:2:3 with points pt 5 ps 0.4 lc variable notitle

# 第二幅图
set title "loss"
set xlabel "Iteration"
set ylabel "Loss value"
set xrange [0:200]
set yrange [0:2]
plot './lossData.txt' using 1:2 with linespoints pt 7 ps 0.4 notitle

unset multiplot
set output  # 关闭文件输出