import os

try:
    import graphviz
except ImportError:
    print("Thiếu thư viện 'graphviz'. Vui lòng chạy: pip install graphviz")
    exit(1)

def generate_slip_diagram(output_path="outputs/slip_pipeline_diagram"):
    """
    Tạo ảnh sơ đồ khối pipeline Real-time Slip Detection
    """
    dot = graphviz.Digraph(comment='Real-time Slip Pipeline', format='png')
    dot.attr(rankdir='TB')  # Tổng thể từ trên xuống dưới
    
    # Thiết lập style chung cho các khối to rõ ràng
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial', fontsize='16', margin='0.4,0.2')

    # --- NHÁNH 1: Live Webcam Flow ---
    with dot.subgraph() as live:
        live.attr(rank='same')
        live.node('Webcam', 'webcam', fillcolor='#E8E8E8', shape='note')
        live.node('Gray', 'grayscale')
        live.node('Pre', 'preprocess')
        live.node('DetLive', 'detect')
        
        live.edge('Webcam', 'Gray')
        live.edge('Gray', 'Pre')
        live.edge('Pre', 'DetLive')

    # --- NHÁNH 2: Reference Flow ---
    with dot.subgraph() as ref:
        ref.attr(rank='same')
        ref.node('Btn', "[nhấn 'r']", shape='cds', fillcolor='#FFDAB9')
        ref.node('RefDet', 'reference → detect', fillcolor='#90EE90')
        ref.node('RefImg', 'reference đã lưu', fillcolor='#E8E8E8', shape='note')

    # Kết nối phím với hành động
    dot.edge('Btn', 'RefDet')
    dot.edge('Btn', 'RefImg')

    # --- NHÁNH 3: Tracking và Phân tích ---
    dot.node('LK', 'LK track', fillcolor='#FFD700', shape='ellipse')
    dot.node('Mask', 'validity mask', fillcolor='#FFB6C1', shape='diamond')
    
    with dot.subgraph() as slip_logic:
        slip_logic.attr(rank='same')
        slip_logic.node('Buf', 'history buffer\n(5 frames)', fillcolor='#E6E6FA')
        slip_logic.node('Slip', 'slip_detector', fillcolor='#FF6347') # Màu đỏ nhạt cho báo động

    dot.node('Vis', 'hiển thị\n(arrows + slip status)', fillcolor='#FFFACD', shape='note')

    # --- NỐI LOGIC ---
    # Đầu vào LK Track
    dot.edge('DetLive', 'LK')
    dot.edge('RefDet', 'LK')
    dot.edge('RefImg', 'LK')

    # Sau LK
    dot.edge('LK', 'Mask')
    dot.edge('Mask', 'Buf')
    dot.edge('Buf', 'Slip')

    # Hiển thị
    dot.edge('Slip', 'Vis')
    dot.edge('Mask', 'Vis') # Có thể nối mask sang hiển thị báo hiệu marker nào còn giữ
    
    # Lưu kết quả
    try:
        dot.render(output_path, view=False, cleanup=True)
        print(f"[Thành công] Đã tạo ảnh sơ đồ Slip Pipeline tại: {output_path}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        print("Lỗi: Không tìm thấy engine của Graphviz. Hãy cài đặt hệ thống: sudo apt-get install graphviz")

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    generate_slip_diagram()
