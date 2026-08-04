import os
import urllib.request
import urllib.parse
import json
import re
import time
import sys

def report_hook(count, block_size, total_size):
    """
    urllib.request.urlretrieve 的回调函数，用于动态打印下载进度条
    """
    if total_size > 0:
        percent = int(count * block_size * 100 / total_size)
        downloaded_mb = count * block_size / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        # 限制百分比在 100 以内
        percent = min(100, percent)
        sys.stdout.write(f"\r      下载进度: [{percent:3d}%] {downloaded_mb:.1f}MB / {total_mb:.1f}MB")
        sys.stdout.flush()
    else:
        # 当获取不到总大小时，只打印已下载量
        downloaded_mb = count * block_size / (1024 * 1024)
        sys.stdout.write(f"\r      下载中: {downloaded_mb:.1f}MB")
        sys.stdout.flush()

def main():
    repo = "junmoxiao-cloud/2026BMI-"
    url = f"https://api.github.com/repos/{repo}/git/trees/main?recursive=1"
    
    print("正在获取 GitHub 仓库文件树...")
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            data = json.loads(response.read().decode())
    except Exception as e:
        print(f"获取数据失败: {e}")
        return

    # 提取所有包含 checkpoint_test_best.pth 的路径
    ckpt_paths = sorted([p['path'] for p in data.get('tree', []) if 'checkpoint_test_best.pth' in p['path']])
    
    base_local_dir = r"d:\NEOschool\2026BMI-\results"
    
    download_tasks = []
    
    for path in ckpt_paths:
        # 解析 mode (intra / inter)
        mode_match = re.search(r'(intra|inter)', path)
        if not mode_match:
            continue
        mode = mode_match.group(1)
        
        # 解析 subject (sub-01 到 sub-10)
        sub_match = re.search(r'(sub-\d{2})', path)
        if not sub_match:
            continue
        sub = sub_match.group(1)
        
        # 解析 seed 
        # 情况1: results/inter-subjects/...-seed3407/...
        seed_match = re.search(r'seed(\d+)', path)
        if seed_match:
            seed = seed_match.group(1)
        else:
            # 情况2: NeuroBridge/1234-inter-subjects/...
            seed_match2 = re.search(r'(?:NeuroBridge/)?(\d+)-(?:inter|intra)', path)
            if seed_match2:
                seed = seed_match2.group(1)
            else:
                continue
                
        # 构建标准化、修复后的本地路径
        # 目标格式: d:\NEOschool\2026BMI-\results\intra-subjects-seed42\sub-01-seed42\checkpoint_test_best.pth
        target_dir = os.path.join(base_local_dir, f"{mode}-subjects-seed{seed}", f"{sub}-seed{seed}")
        target_path = os.path.join(target_dir, 'checkpoint_test_best.pth')
        
        # 编码 URL，使用 blob URL 并追加 ?raw=true 来解析 LFS 文件
        raw_url = f"https://github.com/{repo}/blob/main/{urllib.parse.quote(path)}?raw=true"
        
        download_tasks.append({
            'remote_path': path,
            'raw_url': raw_url,
            'local_path': target_path,
            'desc': f"{mode} | {sub} | seed {seed}"
        })

    # 去重处理：部分 subject 存在带不同时间戳的多次运行记录。
    # 由于 ckpt_paths 已按字母(时间戳)排序，后面的最新运行结果会自然覆盖前面的记录。
    task_dict = {task['local_path']: task for task in download_tasks}
    
    print(f"在 GitHub 上共解析出 {len(task_dict)} 个唯一有效的权重文件 (预期 100 个)。")
    
    downloaded_count = 0
    skipped_count = 0
    
    for local_path, task in task_dict.items():
        mode = task['desc'].split(' | ')[0].strip()
        
        # 真实的 intra 模型权重文件在 77 MB 左右，inter 模型在 44 MB 左右。
        # 分别设定安全阈值：intra > 75MB，inter > 42MB
        threshold = 75000000 if mode == 'intra' else 42000000
        
        if os.path.exists(local_path):
            if os.path.getsize(local_path) > threshold:
                skipped_count += 1
                continue
            else:
                print(f"检测到 {local_path} 大小异常 ({os.path.getsize(local_path)} 字节，预期应大于 {threshold} 字节)，准备重新下载实际权重...")
                os.remove(local_path)
            
        print(f"正在下载: {task['desc']}\n  -> 存储至: {local_path}\n  <- 来源于: {task['remote_path']}")
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        try:
            # 传入 reporthook 以显示进度条
            urllib.request.urlretrieve(task['raw_url'], local_path, reporthook=report_hook)
            sys.stdout.write("\n") # 进度条结束后换行
            downloaded_count += 1
            time.sleep(0.1) # 稍微延迟，避免触发 GitHub 速率限制
        except Exception as e:
            sys.stdout.write("\n") # 进度条结束后换行
            print(f"下载失败 {task['raw_url']}: {e}")
            
    print(f"\n=================================")
    print(f"同步完成！")
    print(f"新增下载: {downloaded_count} 个文件")
    print(f"本地已存: {skipped_count} 个文件 (已跳过)")
    print(f"=================================")

if __name__ == '__main__':
    main()
