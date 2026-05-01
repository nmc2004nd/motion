"""Polynomial regression network cho force estimation.

Cách tiếp cận:
    displacement field (N markers) → scalar features (~10D)
    → standardize → polynomial expansion (degree d)
    → linear (hoặc MLP nhỏ) → scalar force.

So với force_model (PointNet) và force_cnn (ResNet18 end-to-end), force_poly
là baseline tường minh, ít tham số (~50–200), train nhanh, dễ kiểm tra hệ số.
"""
