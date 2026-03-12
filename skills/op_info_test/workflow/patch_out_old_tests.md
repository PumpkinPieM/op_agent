<a id="patch-out-old-tests-steps"></a>
`op_database.py`中包含已有用例和新增用例，为了单独测试新增用例，需要新建仅包含新增接口的xxx_db list，覆盖（非删除）已有的database注册。

这一步不要使用lint工具进行代码格式修改。

示例：
```
@@ -7848,6 +7969,7 @@ other_op_db = [
     'mint.nn.functional.linear',
     'mint.nn.Linear',
     'mint.nn.Conv1d',
+    'mint.nn.Conv2d',
     'Tensor.masked_scatter',
     'Tensor.masked_scatter_',
     'Tensor.add_',
```

该commit在`other_op_db`中新增'mint.nn.Conv2d'接口测试。为了覆盖已有注册应在注册后新建other_op_db：

```
other_op_db = [
    ...
]
other_op_db = ['mint.nn.Conv2d']
```

覆盖已有注册，确保后续测试验证仅测试新增接口。

对于未新增接口的xxx_db，使用空list覆盖:

```
binary_op_db = [
    ...
]
binary_op_db = []
```

最后，确保所有xxx_db均被处理。修改完成后生成git commit，commit信息为"op_info_test: patch out old tests"。
