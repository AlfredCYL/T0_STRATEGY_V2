import matplotlib.pyplot as plt
import numpy as np
import io
import matplotlib.image as mpimg    

# 配置matplotlib支持中文显示
def setup_chinese_font():
    """设置matplotlib支持中文显示"""
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    
    # 如果上述字体都不可用，尝试使用系统默认字体
    try:
        import matplotlib.font_manager as fm
        # 查找系统中的中文字体
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_fonts = [f for f in fonts if any(keyword in f for keyword in ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi'])]
        if chinese_fonts:
            plt.rcParams['font.sans-serif'] = chinese_fonts[:1] + ['DejaVu Sans']
    except:
        pass

# 初始化中文字体设置
setup_chinese_font()

def combine_subplots(figlist, cols_per_row=3, figsize=(15, 10)):
    """
    通用函数：将多个图像对象绘制在一个大图中
    
    参数:
    figlist: list of matplotlib figure objects - 图像对象列表
    cols_per_row: int - 每行显示的图像数量，默认为3
    figsize: tuple - 整体图像大小，默认为(15, 10)
    """
    if not figlist:
        print("图像列表为空，无法绘制")
        return
    
    n_figs = len(figlist)
    n_rows = int(np.ceil(n_figs / cols_per_row))
    
    # 创建主图
    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=figsize)
    
    # 确保axes是二维数组
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    elif cols_per_row == 1:
        axes = axes.reshape(-1, 1)
    
    # 绘制每个子图
    for i, subfig in enumerate(figlist):
        row = i // cols_per_row
        col = i % cols_per_row
        
        # 直接将原图的canvas内容复制到新位置
        target_ax = axes[row, col]
        
        # 获取原图的第一个axes
        source_axes = subfig.get_axes()
        if source_axes:
            buf_io = io.BytesIO()
            subfig.savefig(buf_io, format='png', bbox_inches='tight', dpi=100)
            buf_io.seek(0)
            
            # 读取图像数据
            img = mpimg.imread(buf_io)
            
            # 显示图像
            target_ax.imshow(img)
            target_ax.axis('off')
    
    # 隐藏多余的子图
    for i in range(n_figs, n_rows * cols_per_row):
        row = i // cols_per_row
        col = i % cols_per_row
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def test_combine_subplots():
    """测试combine_subplots函数的功能"""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 创建测试数据
    x = np.linspace(0, 10, 100)
    
    # 创建几个测试子图
    figlist = []
    
    # 子图1: 正弦波
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(x, np.sin(x), 'b-', label='sin(x)')
    ax1.set_title('正弦函数')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.legend()
    ax1.grid(True)
    figlist.append(fig1)
    
    # 子图2: 余弦波
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    ax2.plot(x, np.cos(x), 'r--', label='cos(x)')
    ax2.set_title('余弦函数')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    ax2.grid(True)
    figlist.append(fig2)
    
    # 子图3: 指数函数
    fig3, ax3 = plt.subplots(figsize=(6, 4))
    ax3.plot(x, np.exp(-x/5), 'g:', label='exp(-x/5)')
    ax3.set_title('指数衰减函数')
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.legend()
    ax3.grid(True)
    figlist.append(fig3)
    
    # 子图4: 多条线
    fig4, ax4 = plt.subplots(figsize=(6, 4))
    ax4.plot(x, x**2, 'purple', label='x²')
    ax4.plot(x, x**3/10, 'orange', label='x³/10')
    ax4.set_title('多项式函数')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.legend()
    ax4.grid(True)
    figlist.append(fig4)
    
    print("测试combine_subplots函数...")
    print(f"创建了{len(figlist)}个子图")
    
    # 测试不同的布局
    print("\n测试2x2布局:")
    combine_subplots(figlist, cols_per_row=2, figsize=(12, 10))
    
    print("\n测试1x4布局:")
    combine_subplots(figlist, cols_per_row=4, figsize=(16, 4))
    
    print("\n测试4x1布局:")
    combine_subplots(figlist, cols_per_row=1, figsize=(6, 16))
    
    # 测试只有3个子图的情况
    print("\n测试3个子图的2x2布局:")
    combine_subplots(figlist[:3], cols_per_row=2, figsize=(12, 8))
    
    # 关闭原始图形以节省内存
    for fig in figlist:
        plt.close(fig)
    
    print("测试完成!")

if __name__ == "__main__":
    test_combine_subplots()
