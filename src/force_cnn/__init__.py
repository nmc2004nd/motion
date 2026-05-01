"""End-to-end CNN force regressor: cặp ảnh (reference, deformed) → scalar force.

Khác với src.force_model (PointNet trên displacement field từ LK tracking),
module này train trực tiếp một CNN trên pixel — không cần detection/tracking
ở thời điểm inference.
"""
