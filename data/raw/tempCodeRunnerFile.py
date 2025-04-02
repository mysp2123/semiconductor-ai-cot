import pandas as pd
import numpy as np
import random

# Định nghĩa các tên vật liệu và cấu trúc tinh thể giả
materials = ['CuO', 'Si', 'Ge', 'GaAs', 'CuMnSnO', 'ZnO']
structures = ['Monoclinic', 'Cubic', 'Tetragonal', 'Hexagonal', 'Orthorhombic']

# Hàm tạo giá trị giả cho điện trở, bandgap và nhãn (target)
def generate_fake_data_with_labels(num_samples):
    data = []
    
    for _ in range(num_samples):
        # Chọn ngẫu nhiên vật liệu và cấu trúc
        material = random.choice(materials)
        structure = random.choice(structures)
        
        # Tạo giá trị giả cho các thuộc tính vật liệu
        resistivity = round(np.random.uniform(0.01, 1.0), 4)  # Giả sử giá trị điện trở suất trong khoảng từ 0.01 đến 1.0
        bandgap = round(np.random.uniform(1.0, 3.5), 2)  # Bandgap giả trong khoảng 1.0 - 3.5 eV
        
        # Gán nhãn (target) dựa trên một quy tắc giả định
        # Ví dụ: Nếu bandgap > 2.0 và resistivity < 0.5 thì target = 1, ngược lại target = 0
        target = 1 if bandgap > 2.0 and resistivity < 0.5 else 0
        
        # Tạo một dòng dữ liệu
        data.append({
            'material_name': material,
            'resistivity': resistivity,
            'bandgap': bandgap,
            'structure': structure,
            'target': target
        })
    
    return pd.DataFrame(data)

# Hàm lưu dữ liệu vào file CSV
def save_data_to_csv(df, filename='fake_materials_data_with_labels.csv'):
    df.to_csv(filename, index=False)
    print(f"Data saved to {filename}")

# Tạo bộ dữ liệu giả với 1000 mẫu và gán nhãn
fake_data_with_labels = generate_fake_data_with_labels(1000)

# Lưu bộ dữ liệu vào CSV
save_data_to_csv(fake_data_with_labels)
