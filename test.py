#!/usr/bin/env python3
"""
简单的测试脚本
"""

def main():
    print("这是一个测试脚本")
    print("当前时间:", __import__('datetime').datetime.now())
    print("测试完成！")

if __name__ == "__main__":
    main()
