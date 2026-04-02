# 图像特征检测与匹配实验平台

这是一个无需后端、打开 `index.html` 即可运行的交互式 Web 应用，覆盖以下实验要求：

1. Canny 边缘检测，并展示梯度图、非最大值抑制结果和最终边缘。
2. Harris 与 SIFT 风格特征点检测，并在原图上可视化特征点圆形区域。
3. 两幅图像的匹配流程可视化：特征点检测、描述、初始匹配、RANSAC、变换与对齐。
4. 多幅图像全景拼接，并支持 `overlay`、`average`、`linear feather` 三种 blending 对比。
5. 交互式 Web 形式，适合课程展示与部署到静态托管平台。

## 运行方式

直接双击 [index.html](C:\Users\27260\Desktop\研一第二学期课程\A3\index.html) 即可。

如果浏览器限制本地文件访问，建议在当前目录启动一个静态文件服务：

```powershell
cd C:\Users\27260\Desktop\研一第二学期课程\A3
python -m http.server 8000
```

然后在浏览器访问 `http://localhost:8000`。

## 项目结构

- [index.html](C:\Users\27260\Desktop\研一第二学期课程\A3\index.html)：界面结构
- [styles.css](C:\Users\27260\Desktop\研一第二学期课程\A3\styles.css)：视觉样式
- [app.js](C:\Users\27260\Desktop\研一第二学期课程\A3\app.js)：核心算法与交互逻辑
- [vision_core.py](C:\Users\27260\Desktop\研一第二学期课程\A3\vision_core.py)：课程提交可附带的 Python 核心源码
- [report_template.md](C:\Users\27260\Desktop\研一第二学期课程\A3\report_template.md)：实验报告模板
- [prompt.txt](C:\Users\27260\Desktop\研一第二学期课程\A3\prompt.txt)：实验使用 prompt 文本

## 使用建议

- 单图建议选择纹理丰富、边缘清晰的图像。
- 两图匹配建议使用有明显重叠区域的同一场景图像。
- 全景拼接建议上传 3 张及以上、相邻图像有 30% 以上重叠。
- 由于项目采用纯前端实现，SIFT 部分为课程展示友好的 SIFT 风格实现，用于完整呈现尺度空间、描述子和匹配流程。
