import os

try:
    import graphviz
except ImportError:
    print("Thiếu thư viện 'graphviz'. Vui lòng chạy: pip install graphviz")
    exit(1)

def apply_common_style(dot):
    dot.attr(rankdir='TB', nodesep='0.5', ranksep='0.6')
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='white', fontname='Arial', fontsize='14', margin='0.3,0.15')

def generate_force_poly_diagram(out_dir):
    dot = graphviz.Digraph(comment='Force Poly Pipeline', format='png')
    apply_common_style(dot)
    
    # Inputs
    dot.node('Ref', 'Ảnh tham chiếu\n(ref.jpg)', fillcolor='#E8E8E8', shape='note')
    dot.node('Def', 'Ảnh biến dạng\n(frame.jpg)', fillcolor='#E8E8E8', shape='note')
    
    # CV
    dot.node('Blob', 'Blob Detector\n→ ref_pts (N, 2)', fillcolor='#B0E0E6')
    dot.node('LK', 'Lucas-Kanade Tracking\n→ disp (N, 2), valid (N,)', fillcolor='#B0E0E6')
    
    # Feature Engineering
    dot.node('Feat', 'Feature Engineering\n(9 scalar features)', fillcolor='#FFE4B5')
    
    # Model
    with dot.subgraph(name='cluster_model') as c:
        c.attr(style='dashed', color='gray', label='Polynomial Model', fontsize='16')
        c.node('Std', 'Standardize (mean/std)', fillcolor='#E6E6FA')
        c.node('Poly', 'Polynomial Expand\n(degree d)', fillcolor='#E6E6FA')
        c.node('Head', 'Linear / MLP Head', fillcolor='#E6E6FA')
    
    # Output
    dot.node('Out', 'Force (scalar)', shape='ellipse', fillcolor='#98FB98', style='filled,bold')
    
    # Edges
    dot.edge('Ref', 'Blob')
    dot.edge('Blob', 'LK')
    dot.edge('Def', 'LK')
    dot.edge('LK', 'Feat')
    
    dot.edge('Feat', 'Std')
    dot.edge('Std', 'Poly')
    dot.edge('Poly', 'Head')
    dot.edge('Head', 'Out')
    
    path = os.path.join(out_dir, 'force_poly_diagram')
    dot.render(path, view=False, cleanup=True)
    print(f"Đã tạo: {path}.png")


def generate_force_model_diagram(out_dir):
    dot = graphviz.Digraph(comment='Force Model Pipeline', format='png')
    apply_common_style(dot)
    
    # Inputs
    dot.node('Raw', 'Raw Session Data\n(frames, force_log, ref)', fillcolor='#E8E8E8', shape='folder')
    dot.node('Prep', 'offline_prepare.py\n(Đồng bộ hóa & Tracking)', fillcolor='#FFDAB9')
    dot.node('Cache', '.npz Cache\nref_pts, disp, valid, force', fillcolor='#E8E8E8', shape='cylinder')
    
    # Model Input
    dot.node('Input', 'Point Features (N_max, 4)\n[x_norm, y_norm, dx_norm, dy_norm]', fillcolor='#FFE4B5')
    
    # Model
    with dot.subgraph(name='cluster_model') as c:
        c.attr(style='dashed', color='gray', label='PointNet (ForceNet)', fontsize='16')
        c.node('Shared', 'Shared MLP (Conv1d)\n4 → 64 → 128 → 256', fillcolor='#E6E6FA')
        c.node('Pool', 'Masked Pooling\n(Max + Mean Pooling) → Concat (512)', fillcolor='#DDA0DD')
        c.node('Head', 'Head MLP\nLinear → ReLU → Drop → Linear', fillcolor='#E6E6FA')
    
    # Output
    dot.node('Out', 'Force (B,)', shape='ellipse', fillcolor='#98FB98', style='filled,bold')
    
    # Edges
    dot.edge('Raw', 'Prep')
    dot.edge('Prep', 'Cache')
    dot.edge('Cache', 'Input')
    
    dot.edge('Input', 'Shared', label=' Transpose (B, 4, N_max)', fontsize='12')
    dot.edge('Shared', 'Pool', label=' (B, 256, N_max)', fontsize='12')
    dot.edge('Pool', 'Head', label=' (B, 512)', fontsize='12')
    dot.edge('Head', 'Out')
    
    path = os.path.join(out_dir, 'force_model_diagram')
    dot.render(path, view=False, cleanup=True)
    print(f"Đã tạo: {path}.png")


def generate_force_cnn_diagram(out_dir):
    dot = graphviz.Digraph(comment='Force CNN Pipeline', format='png')
    apply_common_style(dot)
    
    # Inputs
    with dot.subgraph() as inputs:
        inputs.attr(rank='same')
        inputs.node('Ref', 'ref.jpg', fillcolor='#E8E8E8', shape='note')
        inputs.node('Def', 'frame.jpg', fillcolor='#E8E8E8', shape='note')
    
    # Preprocess
    dot.node('Pre1', 'Grayscale + Resize + Normalize', fillcolor='#B0E0E6')
    dot.node('Pre2', 'Grayscale + Resize + Normalize', fillcolor='#B0E0E6')
    dot.node('Stack', 'Stack Channels\nInput: (B, 2, H, W)', fillcolor='#FFE4B5')
    
    # Model
    with dot.subgraph(name='cluster_model') as c:
        c.attr(style='dashed', color='gray', label='ForceCNN (End-to-End)', fontsize='16')
        
        c.node('Backbone', 'Backbone CNN\n(ResNet18 inflated / SmallCNN)', fillcolor='#E6E6FA')
        c.node('Pool', 'AdaptiveAvgPool2d + Flatten\n→ (B, 512) hoặc (B, 128)', fillcolor='#DDA0DD')
        c.node('Head', 'Head MLP\nLinear → ReLU → Drop → Linear', fillcolor='#E6E6FA')
    
    # Output
    dot.node('Out', 'Force (B,)', shape='ellipse', fillcolor='#98FB98', style='filled,bold')
    
    # Edges
    dot.edge('Ref', 'Pre1')
    dot.edge('Def', 'Pre2')
    dot.edge('Pre1', 'Stack')
    dot.edge('Pre2', 'Stack')
    
    dot.edge('Stack', 'Backbone')
    dot.edge('Backbone', 'Pool')
    dot.edge('Pool', 'Head')
    dot.edge('Head', 'Out')
    
    path = os.path.join(out_dir, 'force_cnn_diagram')
    dot.render(path, view=False, cleanup=True)
    print(f"Đã tạo: {path}.png")

if __name__ == '__main__':
    out_dir = "outputs/model_diagrams"
    os.makedirs(out_dir, exist_ok=True)
    generate_force_poly_diagram(out_dir)
    generate_force_model_diagram(out_dir)
    generate_force_cnn_diagram(out_dir)
    print(f"Hoàn thành! Bạn có thể xem các sơ đồ trong thư mục '{out_dir}'")
