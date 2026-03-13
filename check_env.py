#!/usr/bin/env python3
"""
环境诊断与自动修复工具
======================

自动检测并修复常见的版本冲突问题
"""

import subprocess
import sys
import os

def run_cmd(cmd):
    """执行命令并返回输出"""
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

def check_module(module_name):
    """检查模块是否可导入"""
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'unknown')
        return True, version
    except Exception as e:
        return False, str(e)

def main():
    print("=" * 70)
    print("🔍 Python 环境诊断工具")
    print("=" * 70)
    
    # 1. Python 版本
    print(f"\n【Python 信息】")
    print(f"版本: {sys.version}")
    print(f"路径: {sys.executable}")
    
    # 2. 检查关键包
    print(f"\n【已安装包检查】")
    critical_packages = {
        'torch': 'PyTorch（核心）',
        'torchvision': '计算机视觉',
        'numpy': '数值计算',
        'matplotlib': '可视化'
    }
    
    results = {}
    for pkg, desc in critical_packages.items():
        ok, info = check_module(pkg)
        results[pkg] = (ok, info)
        status = "✓" if ok else "✗"
        print(f"{status} {pkg:15} ({desc:15}): {info}")
    
    # 3. GPU 检查
    print(f"\n【GPU 检查】")
    if results.get('torch', (False, None))[0]:
        import torch
        if torch.cuda.is_available():
            print(f"✓ GPU 可用: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA 版本: {torch.version.cuda}")
            print(f"  cuDNN 版本: {torch.backends.cudnn.version()}")
        else:
            print("⚠ GPU 不可用（使用 CPU）")
    
    # 4. 版本兼容性检查
    print(f"\n【版本兼容性】")
    if results.get('torch', (False, None))[0] and results.get('torchvision', (False, None))[0]:
        import torch
        try:
            import torchvision
            # 尝试导入关键功能
            from torchvision import transforms
            print("✓ torch 与 torchvision 兼容")
        except Exception as e:
            print(f"✗ torch 与 torchvision 不兼容: {e}")
            print("\n【推荐修复】")
            print("方案 1: pip uninstall torchvision -y && pip install --user torchvision==0.19.0 --no-deps")
            print("方案 2: 创建虚拟环境（见下方）")
    
    # 5. pip 配置
    print(f"\n【pip 配置】")
    pip_conf = os.path.expanduser("~/.pip/pip.conf")
    if os.path.exists(pip_conf):
        with open(pip_conf) as f:
            print(f"✓ 已配置镜像源:")
            for line in f:
                if 'index-url' in line:
                    print(f"  {line.strip()}")
    else:
        print("⚠ 未配置镜像源（可能下载慢）")
    
    # 6. 推荐操作
    print("\n" + "=" * 70)
    print("📋 推荐操作")
    print("=" * 70)
    
    all_ok = all(v[0] for v in results.values())
    
    if all_ok:
        print("🎉 环境完好！可以开始训练")
        print("\n快速开始:")
        print("  cd /student/wangp126/ml_learning")
        print("  python3 02_cnn/mnist_demo.py  # 运行演示")
    else:
        print("⚠️ 存在问题，建议：")
        print("\n方案 A: 快速修复（适合急用）")
        print("  pip install --user torch torchvision matplotlib --no-deps")
        
        print("\n方案 B: 创建虚拟环境（推荐）")
        print("  python3 -m venv ~/ml_env")
        print("  source ~/ml_env/bin/activate")
        print("  pip install torch==2.8.0 torchvision==0.19.0 matplotlib")
        
        print("\n方案 C: 使用 conda（最稳定）")
        print("  conda create -n ml python=3.10")
        print("  conda activate ml")
        print("  conda install pytorch torchvision -c pytorch")
    
    print("\n详细指南: cat 环境管理指南.md")
    print("=" * 70)

if __name__ == '__main__':
    main()
