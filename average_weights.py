import os
import torch
import copy

def main():
    # 定义遍历的参数
    modes = ['intra-subjects', 'inter-subjects']
    subjects = [f'sub-{i:02d}' for i in range(1, 11)]
    seeds = [42, 3407, 9999, 2025, 1234]
    
    # 获取目标路径
    base_dir = r"d:\NEOschool\2026BMI-\results"
    
    for mode in modes:
        for subject in subjects:
            state_dicts = []
            valid_seeds = []
            
            # 遍历所有种子收集权重
            for seed in seeds:
                # 真实的目录结构: d:\NEOschool\2026BMI-\results\intra-subjects-seed3407\sub-01-seed3407\checkpoint_test_best.pth
                mode_seed_dir = f"{mode}-seed{seed}"
                sub_seed_dir = f"{subject}-seed{seed}"
                ckpt_path = os.path.join(base_dir, mode_seed_dir, sub_seed_dir, 'checkpoint_test_best.pth')
                
                if not os.path.exists(ckpt_path):
                    print(f"警告：未找到文件 {ckpt_path}")
                    continue
                
                print(f"正在加载：{ckpt_path}")
                # 加载权重到 CPU 以避免显存溢出
                ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
                
                # 提取状态字典 (兼容不同的保存格式)
                if 'state_dict' in ckpt:
                    sd = ckpt['state_dict']
                elif 'model_state_dict' in ckpt:
                    sd = ckpt['model_state_dict']
                else:
                    sd = ckpt  # 假设直接保存了状态字典
                    
                state_dicts.append(sd)
                valid_seeds.append(seed)
                
            if not state_dicts:
                print(f"跳过 {mode}/{subject}，未找到任何可用的权重文件。\n")
                continue
                
            print(f"开始计算 {mode}/{subject} 的平均权重 (共融合 {len(valid_seeds)} 个种子)...")
            
            # 初始化平均权重字典，以第一个有效种子的结构为基础
            avg_state_dict = copy.deepcopy(state_dicts[0])
            
            # 对所有参数张量求平均
            for key in avg_state_dict.keys():
                # 检查是否为张量，并且是浮点数类型（跳过诸如 num_batches_tracked 等整型计数器）
                if isinstance(avg_state_dict[key], torch.Tensor) and avg_state_dict[key].is_floating_point():
                    avg_state_dict[key] = sum(sd[key] for sd in state_dicts) / len(state_dicts)
                # 对于非浮点型参数，直接保留第一个种子的值
            
            # 构建 '-averaged' 目录结构
            # 结果将保存在: d:\NEOschool\2026BMI-\results\intra-subjects-averaged\sub-01\checkpoint_test_best.pth
            output_mode_dir = f"{mode}-averaged"
            output_dir = os.path.join(base_dir, output_mode_dir, subject)
            os.makedirs(output_dir, exist_ok=True)
            
            output_path = os.path.join(output_dir, 'checkpoint_test_best.pth')
            
            # 保存计算后的平均权重
            torch.save({'state_dict': avg_state_dict}, output_path)
            print(f"成功保存平均权重至：{output_path}\n")
            
    print("所有受试者的模型权重平均计算已完成！")

if __name__ == '__main__':
    main()
