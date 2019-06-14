# textclassification-douban
Text classification of Douban Film Critics
豆瓣影评文本分类。数据集使用的网上抓取的豆瓣影评数据集，格式为： 电影名##评分##影评共20万条数据。
使用的框架为keras，模型用到了textrnn，textcnn，rcnn等模型。使用交叉验证，最后用stacking方法集成结果。
