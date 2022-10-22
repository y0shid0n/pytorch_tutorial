import time
import pyautogui as pgui

pgui.position()

for _ in range(72):
    time.sleep(600) # 10分待機
    pgui.click(x=1538, y=285, duration=1) # 指定した座標に1秒かけて移動してクリック
    time.sleep(1) # 1秒待機
    pgui.click(x=669, y=285, duration=1)

