# 07_fastapi

目标：把模型封装成可调用服务。

建议脚本数：10

建议文件：
1. 00_fastapi_basics.py: 路由和入参校验
2. 01_pydantic_schema.py: 请求响应模型
3. 02_single_inference_api.py: 单条推理接口
4. 03_batch_inference_api.py: 批量推理接口
5. 04_model_loading.py: 模型加载与缓存
6. 05_error_handling.py: 异常处理和返回码
7. 06_logging_monitoring.py: 请求日志
8. 07_health_check.py: 健康检查
9. 08_simple_auth.py: 简单鉴权
10. 09_api_project.py: 可运行服务项目

学习重点：
- 服务稳定性
- 输入校验与错误处理
- 推理延迟和吞吐意识

验收标准：
- 本地一条命令启动并完成接口测试
