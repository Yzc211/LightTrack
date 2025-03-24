import numpy as np
import matplotlib.pyplot as plt
import matplotlib.axes._axes as axes
import matplotlib.figure as figure
from matplotlib.backends.backend_pdf import PdfPages
from pylab import mpl

pdf = PdfPages('Params-LaSOT_AUC.pdf')
plt.rc('font',family='Times New Roman')
# 设置正常显示符号
mpl.rcParams["axes.unicode_minus"] = False

fig, ax = plt.subplots()  # type:figure.Figure, axes.Axes
ax.set_title('The Performance $vs.$Params on LaSOT', fontsize=15)
ax.set_xlabel('Model Parameters(MB)', fontsize=15)
ax.set_ylabel('AUC(%) on LaSOT-test', fontsize=15)


trackers = ['LightTrack-M', 'E.T.Track', 'HiT-Small', 'FEAR-XS','Ours-Plan1', 'Ours-Plan2', 'Ours-Plan3']
params = np.array([1.97, 6.98, 11.03, 1.37, 1.87, 2.12, 1.97]) #参数量
params_norm = np.array([1.97, 6.98, 11.03, 1.37, 1.87, 2.12, 1.97]) / 3.89 #除以平均值
performance = np.array([52.2, 58.9, 60.5, 50.1, 59.51, 60.43, 58.55]) #精度

circle_color = ['cornflowerblue', 'deepskyblue',  'turquoise', 'orange', 'r', 'r', 'r']
# Marker size in units of points^2
# volume = (200 * (0.15/params_norm) * (performance/61))  ** 2
volume = (210 * (0.18 / params_norm) * (performance / 60) **2 ) ** 2

#绘制圆和圆心
ax.scatter(params, performance, c=circle_color, s=volume, alpha=0.4)
ax.scatter(params, performance, c=circle_color, s=15, marker='o')

# text
ax.text(params[0]+0.5 , performance[0] + 1.2, trackers[0], fontsize=10, color='k')
ax.text(params[1]-0.5 , performance[1] - 1, trackers[1], fontsize=10, color='k')
ax.text(params[2]-1.4 , performance[2]+ 0.5, trackers[2], fontsize=10, color='k')
ax.text(params[3]+1.1 , performance[3] - 1, trackers[3], fontsize=10, color='k')
ax.text(params[4]+1.2, performance[4]-0.3, trackers[4], fontsize=11, color='k')
ax.text(params[5] +1, performance[5] + 0.8 , trackers[5], fontsize=11, color='k')
ax.text(params[6] -0.3, performance[6] - 2.3, trackers[6], fontsize=11, color='k')

ax.grid(which='major', axis='both', linestyle='-.') # color='r', linestyle='-', linewidth=2
ax.set_xlim(0.2, 11.5)
ax.set_ylim(47.5, 63.5)

# 设置更精细的刻度
ax.set_xticks(np.arange(0, 12, 1))  # X轴刻度从0.2到12，间隔0.5
ax.set_yticks(np.arange(48, 63, 2))  # Y轴刻度从48到62，间隔1

ax.xaxis.set_tick_params(labelsize=15)
ax.yaxis.set_tick_params(labelsize=15)


# plot lines
ystart, yend = ax.get_ylim()
# 将自己的算法用虚线连接
# ax.plot([1.87, 1.87], [ystart, yend], linestyle="--", color='k', linewidth=0.7)
# ax.plot([1.87, 1.97], [59.51,  58.55], linestyle="--", color='r', linewidth=0.7)
# ax.plot([1.97, 2.12], [58.55, 60.43], linestyle="--", color='r', linewidth=0.7)

fig.savefig('params-AUC.svg')


pdf.savefig()
pdf.close()
plt.show()
