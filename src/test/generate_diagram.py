import os

try:
    import graphviz
except ImportError:
    print("Thiếu thư viện 'graphviz'.")
    print("Vui lòng chạy: pip install graphviz")
    print("Và đảm bảo máy bạn đã cài GraphViz (vd: sudo apt install graphviz trên Linux)")
    exit(1)

def generate_pipeline_diagram(output_path="outputs/pipeline_diagram"):
    """
    Tạo ảnh sơ đồ khối thuật toán (pipeline diagram)
    Lưu dưới dạng ảnh PNG.
    """
    # Khởi tạo đồ thị hướng có hướng (LR: Left to Right)
    dot = graphviz.Digraph(comment='Algorithm Pipeline', format='png')
    dot.attr(rankdir='LR') 
    
    # Thiết lập style chung cho các khối, thêm fontsize, margin để khối to hơn
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial', fontsize='16', margin='0.4,0.2')

    # 1. Các node đầu vào (Màu xám)
    dot.node('R', 'reference.jpg', fillcolor='#E8E8E8', shape='note')
    dot.node('D', 'deformed.jpg', fillcolor='#E8E8E8', shape='note')
    
    # 2. Xử lý ảnh (Màu xanh dương nhạt)
    dot.node('P1', 'preprocess\n(vignette, CLAHE)')
    dot.node('P2', 'preprocess\n(vignette, CLAHE)')
    
    # 3. Detection (Màu xanh lá)
    dot.node('Det', 'detect\n(N markers)', fillcolor='#90EE90')
    
    # 4. Tracking & Visualization (Màu cam & vàng)
    dot.node('LK', 'LK track\n(Optical Flow)', fillcolor='#FFD700', shape='ellipse')
    dot.node('Vis', 'visualize', fillcolor='#FFFACD')
    
    # 5. Output
    dot.node('Out', 'outputs/...', shape='folder', fillcolor='#E8E8E8')

    # --- Kết nối các khối (edges) ---
    
    # Nhánh Reference
    dot.edge('R', 'P1')
    dot.edge('P1', 'Det')
    dot.edge('Det', 'LK')
    
    # Nhánh Deformed
    dot.edge('D', 'P2')
    dot.edge('P2', 'LK')
    
    # Kết xuất / Visualization
    dot.edge('LK', 'Vis')
    dot.edge('Vis', 'Out')

    # Lưu kết quả
    try:
        dot.render(output_path, view=False, cleanup=True)
        print(f"[Thành công] Đã tạo ảnh sơ đồ pipeline tại: {output_path}.png")
    except graphviz.backend.execute.ExecutableNotFound as e:
        print("Lỗi: Không tìm thấy engine của Graphviz trên hệ thống.")
        print("Vui lòng cài đặt: sudo apt-get install graphviz (Linux) hoặc cài từ graphviz.org (Windows)")

if __name__ == '__main__':
    # Tạo output folder nếu chưa có
    os.makedirs("outputs", exist_ok=True)
    generate_pipeline_diagram()
