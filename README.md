# JS-CVdasai
http://jscvc.ujs.edu.cn/
初始数据量很少，首先进行数据扩增，除了常见的旋转，镜像，裁剪之类的手段增加了mask的方法这里代码主要参考了https://github.com/maozezhong/CV_ToolBox.git
由于赛题中实时性得分占了百分之四十，考虑使用单阶端模型，尝试了yolo3和retinanet，最后在retinanet在远程测试集得到了74分，13秒处理完所有图片
