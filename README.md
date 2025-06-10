import os
import cv2
import base64
import json
import logging
import time
import queue
import threading
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import psutil
import shutil
import tempfile
import unittest

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("traffic_detection_system.log")
    ]
)
logger = logging.getLogger('MainSystem')

class ImageBatchLoader:
    def __init__(self, data_source, batch_size=10, max_queue_size=50, memory_threshold=80):
        """
        初始化图片批量加载器
        :param data_source: 数据源路径 (本地目录/数据库连接/API端点)
        :param batch_size: 每批图片数量
        :param max_queue_size: 内存队列最大容量
        :param memory_threshold: 内存使用阈值百分比
        """
        self.data_source = data_source
        self.batch_size = batch_size
        self.image_queue = queue.Queue(maxsize=max_queue_size)
        self.stop_signal = False
        self.memory_threshold = memory_threshold
        self.logger = logging.getLogger('ImageLoader')
        
    def validate_path(self):
        """验证数据源路径有效性"""
        if isinstance(self.data_source, str):  # 本地目录
            if not os.path.exists(self.data_source):
                self.logger.error(f"无效的路径: {self.data_source}")
                return False
            if not os.path.isdir(self.data_source):
                self.logger.error(f"路径不是目录: {self.data_source}")
                return False
        # 其他数据源类型验证（数据库、API等）
        return True

    def retrieve_images(self):
        """检索图片文件"""
        image_files = []
        
        if isinstance(self.data_source, str):  # 本地目录
            for root, _, files in os.walk(self.data_source):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_files.append(os.path.join(root, file))
        
        # 排序：按文件名或拍摄时间
        image_files.sort(key=lambda x: os.path.basename(x))
        return image_files

    def filter_images(self, image_files, min_width=640, min_height=480):
        """根据条件筛选图片"""
        filtered = []
        for img_path in image_files:
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"损坏图像: {img_path}")
                    continue
                if img.shape[1] >= min_width and img.shape[0] >= min_height:
                    filtered.append(img_path)
                else:
                    self.logger.warning(f"低分辨率图像: {img_path} ({img.shape[1]}x{img.shape[0]})")
            except Exception as e:
                self.logger.error(f"处理图像错误 {img_path}: {str(e)}")
        return filtered

    def start_loading(self):
        """启动图片加载线程"""
        if not self.validate_path():
            return None
        
        image_files = self.retrieve_images()
        if not image_files:
            self.logger.error("未找到有效图片文件")
            return None
        
        # 筛选图片
        filtered_files = self.filter_images(image_files)
        self.logger.info(f"找到 {len(filtered_files)} 张有效图片，原始 {len(image_files)} 张")
        
        # 启动加载线程
        loader_thread = threading.Thread(target=self._load_images, args=(filtered_files,))
        loader_thread.daemon = True
        loader_thread.start()
        return loader_thread

    def _load_images(self, image_files):
        """内部加载方法，运行在单独线程中"""
        batch = []
        for i, img_path in enumerate(image_files):
            # 检查内存使用情况
            mem_percent = psutil.virtual_memory().percent
            if mem_percent > self.memory_threshold:
                self.logger.warning(f"内存使用过高 ({mem_percent}%), 暂停加载")
                time.sleep(5)
                continue
                
            try:
                img = cv2.imread(img_path)
                if img is None:
                    self.logger.warning(f"无法读取图片: {img_path}")
                    continue
                
                # 收集元数据
                metadata = {
                    "filepath": img_path,
                    "filename": os.path.basename(img_path),
                    "width": img.shape[1],
                    "height": img.shape[0],
                    "channels": img.shape[2],
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(img_path)))
                }
                
                # 转换为Base64编码用于API传输
                _, buffer = cv2.imencode('.jpg', img)
                img_base64 = base64.b64encode(buffer).decode('utf-8')
                
                batch.append({
                    "image_data": img_base64,
                    "metadata": metadata
                })
                
                # 达到批次大小时放入队列
                if len(batch) >= self.batch_size:
                    self.image_queue.put(batch)
                    self.logger.info(f"已加载批次: {i+1}/{len(image_files)}")
                    batch = []
            
            except Exception as e:
                self.logger.error(f"处理图片错误 {img_path}: {str(e)}")
        
        # 处理最后一批
        if batch:
            self.image_queue.put(batch)
            self.logger.info(f"已加载最后一批: {len(batch)} 张图片")
        
        self.stop_signal = True
        self.logger.info("图片加载完成")

    def get_batch(self):
        """从队列获取一批图片"""
        if self.image_queue.empty() and self.stop_signal:
            return None  # 结束信号
        return self.image_queue.get()


class APICaller:
    def __init__(self, api_endpoint, api_key, max_retries=3, retry_delay=1):
        """
        初始化API调用器
        :param api_endpoint: API端点URL
        :param api_key: API密钥
        :param max_retries: 最大重试次数
        :param retry_delay: 重试延迟时间(秒)
        """
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        self.logger = logging.getLogger('APICaller')
    
    def call_api(self, image_data):
        """调用API处理单个图像"""
        payload = {
            "image": image_data["image_data"],
            "model": "traffic-v3",
            "confidence_threshold": 0.65,
            "classes": ["car", "truck", "bus", "person", "bicycle", "motorcycle", 
                       "traffic_light", "stop_sign", "speed_limit"]
        }
        
        for attempt in range(self.max_retries + 1):
            try:
                start_time = time.time()
                response = requests.post(
                    self.api_endpoint, 
                    json=payload, 
                    headers=self.headers,
                    timeout=10
                )
                processing_time = time.time() - start_time
                
                if response.status_code == 200:
                    result = response.json()
                    return {
                        "status": "success",
                        "result": result,
                        "processing_time": processing_time
                    }
                else:
                    error_msg = f"API错误 {response.status_code}: {response.text}"
                    if attempt < self.max_retries:
                        self.logger.warning(f"{error_msg} - 重试 {attempt+1}/{self.max_retries}")
                        time.sleep(self.retry_delay * (2 ** attempt))  # 指数退避
                    else:
                        self.logger.error(error_msg)
                        return {
                            "status": "error",
                            "error": error_msg,
                            "processing_time": processing_time
                        }
            
            except Exception as e:
                error_msg = f"请求异常: {str(e)}"
                if attempt < self.max_retries:
                    self.logger.warning(f"{error_msg} - 重试 {attempt+1}/{self.max_retries}")
                    time.sleep(self.retry_delay * (2 ** attempt))
                else:
                    self.logger.error(error_msg)
                    return {
                        "status": "error",
                        "error": error_msg,
                        "processing_time": 0
                    }
        
        return {
            "status": "error",
            "error": "未知错误",
            "processing_time": 0
        }


class BatchDispatcher:
    def __init__(self, input_queue, api_caller, max_workers=4):
        """
        初始化批次分发器
        :param input_queue: 输入图片队列
        :param api_caller: API调用器实例
        :param max_workers: 最大工作线程数
        """
        self.input_queue = input_queue
        self.api_caller = api_caller
        self.max_workers = max_workers
        self.result_queue = queue.Queue()
        self.active = True
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        self.logger = logging.getLogger('BatchDispatcher')
        self.processed_count = 0

    def start_dispatching(self):
        """启动分发线程"""
        dispatch_thread = threading.Thread(target=self._dispatch_batches)
        dispatch_thread.daemon = True
        dispatch_thread.start()
        return dispatch_thread

    def _dispatch_batches(self):
        """分发批次到工作线程"""
        self.logger.info("分发器启动")
        
        while self.active:
            batch = self.input_queue.get()
            
            if batch is None:  # 结束信号
                self.logger.info("接收到结束信号，停止分发")
                self.active = False
                break
                
            # 提交批次处理任务到线程池
            future = self.thread_pool.submit(self.process_batch, batch)
            future.add_done_callback(self.handle_result)
            
            self.logger.info(f"已分发批次: {len(batch)} 张图片")
            
            # 避免过度消耗CPU
            time.sleep(0.1)
        
        # 关闭线程池
        self.thread_pool.shutdown(wait=True)
        self.logger.info("分发线程已停止")

    def process_batch(self, batch):
        """处理单个批次"""
        results = []
        for img_data in batch:
            try:
                # 调用API处理
                api_result = self.api_caller.call_api(img_data)
                results.append({
                    "metadata": img_data["metadata"],
                    "api_result": api_result
                })
            except Exception as e:
                self.logger.error(f"处理批次失败: {str(e)}")
                results.append({
                    "metadata": img_data["metadata"],
                    "error": str(e)
                })
        return results

    def handle_result(self, future):
        """处理完成的任务结果"""
        try:
            batch_result = future.result()
            self.result_queue.put(batch_result)
            self.processed_count += len(batch_result)
            self.logger.info(f"已处理批次: {len(batch_result)} 项结果 (总计: {self.processed_count})")
        except Exception as e:
            self.logger.error(f"处理结果失败: {str(e)}")

    def stop(self):
        """停止分发器"""
        self.active = False


class JSONResultSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger('JSONResultSaver')
        
    def save(self, results, filename=None):
        """保存结果为JSON格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.json"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 转换为可序列化格式
            serializable_results = []
            for result in results:
                # 确保所有数据都是基本类型
                serializable = {
                    "metadata": result["metadata"],
                }
                
                if "api_result" in result:
                    api_result = result["api_result"]
                    if api_result["status"] == "success":
                        serializable["detections"] = api_result["result"].get("detections", [])
                        serializable["processing_time"] = api_result["processing_time"]
                    else:
                        serializable["error"] = api_result.get("error", "API调用失败")
                elif "error" in result:
                    serializable["error"] = result["error"]
                
                serializable_results.append(serializable)
            
            # 写入文件
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"已保存JSON结果: {output_path}")
            return output_path
        
        except Exception as e:
            self.logger.error(f"保存JSON失败: {str(e)}")
            return None


class CSVResultSaver:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.logger = logging.getLogger('CSVResultSaver')
    
    def save(self, results, filename=None):
        """保存结果为CSV格式"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"detection_results_{timestamp}.csv"
        
        output_path = os.path.join(self.output_dir, filename)
        
        try:
            # 准备数据
            data = []
            for result in results:
                base_entry = {
                    "filename": result["metadata"]["filename"],
                    "width": result["metadata"]["width"],
                    "height": result["metadata"]["height"],
                    "timestamp": result["metadata"]["timestamp"]
                }
                
                if "api_result" in result:
                    api_result = result["api_result"]
                    if api_result["status"] == "success":
                        detections = api_result["result"].get("detections", [])
                        for det in detections:
                            entry = base_entry.copy()
                            entry.update({
                                "object_class": det["class"],
                                "confidence": det["confidence"],
                                "x_min": det["bbox"][0],
                                "y_min": det["bbox"][1],
                                "x_max": det["bbox"][2],
                                "y_max": det["bbox"][3],
                                "processing_time": api_result["processing_time"]
                            })
                            data.append(entry)
                    else:
                        entry = base_entry.copy()
                        entry.update({
                            "object_class": "ERROR",
                            "confidence": 0,
                            "error_message": api_result.get("error", "API调用失败"),
                            "processing_time": api_result.get("processing_time", 0)
                        })
                        data.append(entry)
                elif "error" in result:
                    entry = base_entry.copy()
                    entry.update({
                        "object_class": "ERROR",
                        "confidence": 0,
                        "error_message": result["error"],
                        "processing_time": 0
                    })
                    data.append(entry)
            
            # 创建DataFrame并保存
            if data:
                df = pd.DataFrame(data)
                
                # 重新排序列
                columns = ["filename", "timestamp", "width", "height", 
                          "object_class", "confidence", 
                          "x_min", "y_min", "x_max", "y_max", 
                          "processing_time", "error_message"]
                
                # 只包含存在的列
                df = df.reindex(columns=[col for col in columns if col in df.columns])
                
                df.to_csv(output_path, index=False)
                self.logger.info(f"已保存CSV结果: {output_path}")
                return output_path
            else:
                self.logger.warning("没有有效数据可保存")
                return None
        
        except Exception as e:
            self.logger.error(f"保存CSV失败: {str(e)}")
            return None


class ResultExporter:
    def __init__(self, output_dir, formats=('json', 'csv'), buffer_size=50):
        """
        初始化结果导出器
        :param output_dir: 输出目录
        :param formats: 支持的输出格式
        :param buffer_size: 缓冲区大小
        """
        self.output_dir = output_dir
        self.savers = {}
        self.buffer = []
        self.buffer_size = buffer_size
        
        if 'json' in formats:
            self.savers['json'] = JSONResultSaver(output_dir)
        if 'csv' in formats:
            self.savers['csv'] = CSVResultSaver(output_dir)
        
        self.logger = logging.getLogger('ResultExporter')
    
    def add_result(self, result):
        """添加结果到缓冲区"""
        self.buffer.append(result)
        if len(self.buffer) >= self.buffer_size:
            self.flush_buffer()
    
    def flush_buffer(self):
        """将缓冲区内容保存到文件"""
        if not self.buffer:
            return
        
        # 使用当前日期作为文件名
        date_str = datetime.now().strftime("%Y%m%d")
        
        # 保存所有支持格式
        saved_files = []
        for format, saver in self.savers.items():
            if format == 'json':
                filename = f"results_{date_str}.json"
                output_path = os.path.join(self.output_dir, filename)
                
                # JSON追加保存
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        existing_data = json.load(f)
                    existing_data.extend(self._prepare_json_buffer())
                    with open(output_path, 'w') as f:
                        json.dump(existing_data, f, indent=2)
                else:
                    with open(output_path, 'w') as f:
                        json.dump(self._prepare_json_buffer(), f, indent=2)
                saved_files.append(output_path)
            
            elif format == 'csv':
                filename = f"results_{date_str}.csv"
                output_path = os.path.join(self.output_dir, filename)
                
                # CSV追加保存
                if os.path.exists(output_path):
                    df_existing = pd.read_csv(output_path)
                    df_buffer = pd.DataFrame(self._prepare_csv_buffer())
                    df_combined = pd.concat([df_existing, df_buffer])
                    df_combined.to_csv(output_path, index=False)
                else:
                    df_buffer = pd.DataFrame(self._prepare_csv_buffer())
                    if not df_buffer.empty:
                        df_buffer.to_csv(output_path, index=False)
                        saved_files.append(output_path)
        
        self.logger.info(f"已保存 {len(self.buffer)} 条结果")
        self.buffer = []
        return saved_files
    
    def _prepare_json_buffer(self):
        """准备JSON格式的缓冲区数据"""
        json_data = []
        for result in self.buffer:
            item = {
                "metadata": result["metadata"],
            }
            
            if "api_result" in result:
                api_result = result["api_result"]
                if api_result["status"] == "success":
                    item["detections"] = api_result["result"].get("detections", [])
                    item["processing_time"] = api_result["processing_time"]
                else:
                    item["error"] = api_result.get("error", "API调用失败")
            elif "error" in result:
                item["error"] = result["error"]
            
            json_data.append(item)
        return json_data
    
    def _prepare_csv_buffer(self):
        """准备CSV格式的缓冲区数据"""
        csv_data = []
        for result in self.buffer:
            base_entry = {
                "filename": result["metadata"]["filename"],
                "width": result["metadata"]["width"],
                "height": result["metadata"]["height"],
                "timestamp": result["metadata"]["timestamp"]
            }
            
            if "api_result" in result:
                api_result = result["api_result"]
                if api_result["status"] == "success":
                    detections = api_result["result"].get("detections", [])
                    for det in detections:
                        entry = base_entry.copy()
                        entry.update({
                            "object_class": det["class"],
                            "confidence": det["confidence"],
                            "x_min": det["bbox"][0],
                            "y_min": det["bbox"][1],
                            "x_max": det["bbox"][2],
                            "y_max": det["bbox"][3],
                            "processing_time": api_result["processing_time"]
                        })
                        csv_data.append(entry)
                else:
                    entry = base_entry.copy()
                    entry.update({
                        "object_class": "ERROR",
                        "confidence": 0,
                        "error_message": api_result.get("error", "API调用失败"),
                        "processing_time": api_result.get("processing_time", 0)
                    })
                    csv_data.append(entry)
            elif "error" in result:
                entry = base_entry.copy()
                entry.update({
                    "object_class": "ERROR",
                    "confidence": 0,
                    "error_message": result["error"],
                    "processing_time": 0
                })
                csv_data.append(entry)
        
        return csv_data
    
    def finalize(self):
        """处理剩余缓冲区内容"""
        if self.buffer:
            return self.flush_buffer()
        return []


class TrafficDetectionSystem:
    def __init__(self, config):
        """
        初始化交通检测系统
        :param config: 系统配置字典
        """
        self.config = config
        self.logger = logging.getLogger('TrafficSystem')
        
        # 初始化组件
        self.image_loader = ImageBatchLoader(
            config['data_source'],
            batch_size=config.get('batch_size', 10),
            memory_threshold=config.get('memory_threshold', 80)
        )
        
        self.api_caller = APICaller(
            config['api_endpoint'],
            config['api_key'],
            max_retries=config.get('max_retries', 3)
        )
        
        self.dispatcher = BatchDispatcher(
            self.image_loader.image_queue,
            self.api_caller,
            max_workers=config.get('max_workers', 4)
        )
        
        self.exporter = ResultExporter(
            config['output_dir'],
            formats=config.get('output_formats', ['json', 'csv']),
            buffer_size=config.get('buffer_size', 50)
        )
        
        self.running = False
    
    def start(self):
        """启动系统"""
        if self.running:
            self.logger.warning("系统已在运行中")
            return
        
        self.running = True
        self.logger.info("启动交通检测系统")
        
        # 启动加载器
        loader_thread = self.image_loader.start_loading()
        if not loader_thread:
            self.logger.error("图片加载失败，系统终止")
            return
        
        # 启动分发器
        dispatcher_thread = self.dispatcher.start_dispatching()
        
        # 主处理循环
        try:
            processed_count = 0
            while self.running:
                # 检查加载器是否完成
                if loader_thread.is_alive():
                    time.sleep(1)
                    continue
                
                # 检查分发器是否完成
                if not self.dispatcher.active and self.dispatcher.result_queue.empty():
                    break
                
                # 处理结果
                try:
                    batch_result = self.dispatcher.result_queue.get(timeout=5)
                    for result in batch_result:
                        self.exporter.add_result(result)
                        processed_count += 1
                    
                    self.logger.info(f"已处理 {processed_count} 张图片")
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"处理结果时出错: {str(e)}")
        
        except KeyboardInterrupt:
            self.logger.info("用户中断，停止系统...")
        
        finally:
            # 停止系统
            self.running = False
            
            # 确保分发器停止
            self.dispatcher.stop()
            
            # 等待分发器线程结束
            if dispatcher_thread.is_alive():
                dispatcher_thread.join(timeout=10)
            
            # 处理剩余结果
            while not self.dispatcher.result_queue.empty():
                try:
                    batch_result = self.dispatcher.result_queue.get(timeout=1)
                    for result in batch_result:
                        self.exporter.add_result(result)
                except queue.Empty:
                    break
            
            # 最终保存结果
            saved_files = self.exporter.finalize()
            self.logger.info(f"系统停止，结果保存在: {', '.join(saved_files)}")
            
            # 生成性能报告
            self.generate_performance_report(processed_count)
    
    def generate_performance_report(self, total_images):
        """生成性能报告"""
        report = {
            "system": "Traffic Detection System",
            "version": "1.0",
            "run_date": datetime.now().isoformat(),
            "config": self.config,
            "performance": {
                "total_images": total_images,
                "success_rate": 1.0,  # 实际应用中需要计算
                "avg_processing_time": 0.0  # 实际应用中需要计算
            }
        }
        
        report_path = os.path.join(self.config['output_dir'], "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"性能报告已保存: {report_path}")
    
    def stop(self):
        """停止系统"""
        self.running = False


def main():
    # 系统配置
    config = {
        'data_source': 'path/to/traffic/images',  # 替换为实际图片路径
        'api_endpoint': 'https://api.agentic-detection.com/v1/detect',  # 替换为实际API端点
        'api_key': 'your_api_key_here',  # 替换为实际API密钥
        'output_dir': 'results',
        'batch_size': 8,
        'max_workers': 4,
        'max_retries': 3,
        'memory_threshold': 85,
        'output_formats': ['json', 'csv'],
        'buffer_size': 30
    }
    
    # 创建并启动系统
    system = TrafficDetectionSystem(config)
    system.start()


if __name__ == "__main__":
    main()


# ------------------------ 单元测试模块 ------------------------
class TestTrafficSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 创建临时测试目录
        cls.test_dir = tempfile.mkdtemp()
        cls.image_dir = os.path.join(cls.test_dir, "images")
        cls.output_dir = os.path.join(cls.test_dir, "results")
        os.makedirs(cls.image_dir, exist_ok=True)
        os.makedirs(cls.output_dir, exist_ok=True)
        
        # 创建测试图片
        for i in range(10):
            img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(cls.image_dir, f"test_{i}.jpg"), img)
        
        # 创建损坏图片
        with open(os.path.join(cls.image_dir, "corrupted.jpg"), 'w') as f:
            f.write("This is not a valid image file")
    
    @classmethod
    def tearDownClass(cls):
        # 清理临时目录
        shutil.rmtree(cls.test_dir)
    
    def test_full_system(self):
        """测试完整系统流程"""
        # 模拟API响应
        def mock_api_call(image_data):
            return {
                "status": "success",
                "result": {
                    "detections": [
                        {"class": "car", "confidence": 0.92, "bbox": [100, 120, 300, 280]},
                        {"class": "person", "confidence": 0.85, "bbox": [400, 200, 450, 380]}
                    ]
                },
                "processing_time": 0.25
            }
        
        # 系统配置
        config = {
            'data_source': self.image_dir,
            'api_endpoint': 'https://mock-api.com/detect',
            'api_key': 'test_key',
            'output_dir': self.output_dir,
            'batch_size': 5,
            'max_workers': 2,
            'max_retries': 1,
            'output_formats': ['json', 'csv'],
            'buffer_size': 5
        }
        
        # 创建系统
        system = TrafficDetectionSystem(config)
        
        # 替换API调用器为模拟版本
        system.api_caller.call_api = mock_api_call
        
        # 启动系统
        system_thread = threading.Thread(target=system.start)
        system_thread.start()
        
        # 等待系统完成
        system_thread.join(timeout=30)
        
        # 检查结果文件
        json_files = [f for f in os.listdir(self.output_dir) if f.endswith('.json')]
        csv_files = [f for f in os.listdir(self.output_dir) if f.endswith('.csv')]
        
        self.assertTrue(len(json_files) > 0, "未生成JSON结果文件")
        self.assertTrue(len(csv_files) > 0, "未生成CSV结果文件")
        
        # 验证JSON文件内容
        with open(os.path.join(self.output_dir, json_files[0])) as f:
            json_data = json.load(f)
            self.assertGreaterEqual(len(json_data), 10)  # 10张有效图片
        
        # 验证CSV文件内容
        csv_path = os.path.join(self.output_dir, csv_files[0])
        df = pd.read_csv(csv_path)
        self.assertGreaterEqual(len(df), 20)  # 10张图片，每张2个检测结果


# 运行单元测试
if __name__ == "__main__":
    # 运行主系统
    # main()
    
    # 运行单元测试
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
