import os
import shutil
import subprocess
import sys

def download_web_to_app():
    print("🚀 准备下载 Web-to-App 源码仓库...")
    base_dir = os.path.abspath(os.path.dirname(__file__))
    source_dir = os.path.join(base_dir, "source_code")
    
    repo_url = "https://github.com/shiahonb777/web-to-app.git"
    
    # 如果目录已存在，检查是否为空或是之前我们留下的占位文件
    if os.path.exists(source_dir):
        print("[*] 正在清理旧的占位文件或目录...")
        shutil.rmtree(source_dir)
        
    print(f"[*] 正在克隆仓库: {repo_url} -> {source_dir}")
    try:
        subprocess.run(["git", "clone", repo_url, source_dir], check=True)
        print("\n[+] 源码克隆成功！")
        print(f"[+] 路径: {source_dir}")
        print("[+] 你现在可以阅读原版源码并进行重构了。")
    except subprocess.CalledProcessError as e:
        print(f"\n[-] Git 克隆失败，错误信息: {e}")
        print("[-] 提示: 请检查网络连接、Git是否安装，或确认该仓库是否存在且公开。")
        sys.exit(1)

if __name__ == "__main__":
    download_web_to_app()
