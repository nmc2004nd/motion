import os

try:
    import graphviz
except ImportError:
    print("Thiếu thư viện 'graphviz'. Vui lòng chạy: pip install graphviz")
    exit(1)

def generate_collection_diagram(output_path="outputs/collection_thread_diagram"):
    """
    Tạo ảnh sơ đồ khối kiến trúc đa luồng (Multi-threading) cho quá trình thu thập dữ liệu
    """
    dot = graphviz.Digraph(comment='Data Collection Threads Pipeline', format='png')
    dot.attr(rankdir='LR')  # Left to Right
    
    # Thiết lập style chung cho các khối to rõ ràng
    dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial', fontsize='16', margin='0.4,0.2')

    # --- CÁC THREAD ĐẦU VÀO ---
    dot.node('CamT', 'Camera thread\n(poll cv2)', fillcolor='#B0E0E6')
    dot.node('ImaT', 'Imada thread\n(serial 19200)', fillcolor='#B0E0E6')
    dot.node('ArdT', 'Arduino thread\n(serial 115200)', fillcolor='#B0E0E6')

    # --- CÁC QUEUE (Hàng đợi) ---
    # Chỉnh hình dáng thành giống dạng bộ nhớ / kho lưu trữ
    dot.attr('node', shape='cylinder', fillcolor='#FFE4B5', margin='0.2,0.2')
    dot.node('QCam', 'frame_queue\n(maxsize=200)')
    dot.node('QIma', 'force_queue\n(maxsize=500)')
    dot.node('QArd', 'motor_queue')

    # Căn chỉnh để các Queue cùng nằm trên 1 cột dọc (rank='same' sẽ căn thẳng hàng)
    with dot.subgraph() as queues:
        queues.attr(rank='same')
        queues.node('QCam')
        queues.node('QIma')
        queues.node('QArd')

    # --- WRITER THREAD & OUTPUT ---
    dot.attr('node', shape='box', style='rounded,filled,bold', fillcolor='#FFDAB9', margin='0.4,0.2')
    dot.node('Writer', 'Writer thread\n(block tới\nrecording_event)')
    
    dot.attr('node', shape='folder', fillcolor='#E8E8E8', margin='0.3,0.3')
    dot.node('Out', 'Ghi ra đĩa:\n1 jpg + 3 csv')

    # --- NỐI CÁC LUỒNG DỮ LIỆU ---
    # Thread -> Queue
    dot.edge('CamT', 'QCam')
    dot.edge('ImaT', 'QIma')
    dot.edge('ArdT', 'QArd')

    # Queue -> Writer
    dot.edge('QCam', 'Writer')
    dot.edge('QIma', 'Writer')
    dot.edge('QArd', 'Writer')

    # Writer -> Output
    dot.edge('Writer', 'Out', style='bold', color='red')

    # Lưu kết quả
    try:
        dot.render(output_path, view=False, cleanup=True)
        print(f"[Thành công] Đã tạo ảnh sơ đồ Kiến trúc Threading tại: {output_path}.png")
    except graphviz.backend.execute.ExecutableNotFound:
        print("Lỗi: Không tìm thấy engine của Graphviz. Hãy cài đặt hệ thống: sudo apt-get install graphviz")

if __name__ == '__main__':
    os.makedirs("outputs", exist_ok=True)
    generate_collection_diagram()
