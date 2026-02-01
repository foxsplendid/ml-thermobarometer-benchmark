# V6 架构重构变更说明

**日期**：2026-02-01  
**版本**：V6

---

## 一、变更概览

本次重构聚焦于**代码工程质量提升**，包括配置集中化、日志系统引入、测试框架建立、接口文档增强，以及多项 Minor/Major 问题修复。

### 变更统计

| 类别 | 数量 |
|------|------|
| 新增文件 | 9 |
| 修改文件 | 13 |
| 新增测试用例 | 87 |
| 移除冗余代码行 | ~200 |

---

## 二、新增文件

### 1. `config.py` - 集中配置管理

**位置**：项目根目录

**功能**：
- 数据类配置（`DataConfig`, `CVConfig`, `OutputConfig`, `ModelDefaults`）
- 版本信息收集（`get_version_info()`）
- 向后兼容接口（`get_legacy_config()`）
- YAML 配置文件覆盖支持

**使用示例**：
```python
from config import CONFIG, get_version_info

# 访问配置
print(CONFIG.cv.n_splits)       # 10
print(CONFIG.data.path)         # input.csv 路径

# 获取版本信息
version = get_version_info()
print(version['git_commit'])    # Git 提交哈希
print(version['dependencies'])  # 依赖版本
```

---

### 2. `src/logger.py` - 统一日志模块

**功能**：
- 控制台输出（INFO 级别，带颜色）
- 文件输出（DEBUG 级别，保存到 `results/logs/`）
- Windows ANSI 颜色支持
- 实验日志便捷函数

**使用示例**：
```python
from src.logger import get_logger, log_experiment_start

logger = get_logger(__name__)
logger.info("开始处理...")
logger.warning("警告信息")

# 实验专用
log_experiment_start("E07", {"model": "ert", "data": "augmented"})
```

---

### 3. `tests/` 目录 - 测试框架

**新增文件**：
| 文件 | 测试内容 | 用例数 |
|------|----------|--------|
| `conftest.py` | pytest fixtures | - |
| `test_data_modules.py` | M1 数据模块 | 16 |
| `test_model_modules.py` | M2 模型模块 | 18 |
| `test_correction_modules.py` | M3 校正模块 | 12 |
| `test_protocol.py` | Pipeline/CV | 9 |
| `test_splitters.py` | P-T 划分工具 | 15 |
| `test_metrics.py` | 指标计算 | 18 |

**运行测试**：
```bash
# 运行所有测试
pytest tests/ -v

# 运行单个模块测试
pytest tests/test_data_modules.py -v

# 带覆盖率
pytest tests/ -v --cov=src
```

---

## 三、问题修复

### Major 问题

#### M2. AugmentedDataModule 随机种子问题

**问题**：同一实例多次调用 `fit_transform()` 使用相同随机种子

**修复**：添加调用计数器 `_fit_count`

```python
# 修改前
rng = np.random.RandomState(self.random_seed)

# 修改后
effective_seed = self.random_seed + self._fit_count
self._fit_count += 1
rng = np.random.RandomState(effective_seed)
```

**影响**：`src/data_modules.py`

---

#### M4. StratifiedCVProtocol 分层标签警告

**问题**：`stratify_labels=None` 时静默降级为普通 KFold

**修复**：添加警告日志

```python
if stratify_labels is None:
    logger.warning(
        "stratify_labels=None: 使用普通 KFold 而非 StratifiedKFold，"
        "可能导致 CV 折间分布不平衡"
    )
```

**影响**：`src/protocol.py`

---

### Minor 问题

#### m1. apply_seed 函数重复定义

**修复**：提取为模块级函数 `_apply_seed()`

**影响**：`src/protocol.py`（删除 2 处重复定义）

---

#### m2. viz.py 导入保护

**修复**：添加 matplotlib/seaborn 导入保护

```python
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_PLOTTING = True
except ImportError:
    HAS_PLOTTING = False
```

**影响**：`src/viz.py`

---

#### m4. interfaces.py 前向声明

**修复**：使用 `TYPE_CHECKING` 模式

```python
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .protocol import Pipeline
```

**影响**：`src/interfaces.py`

---

## 四、代码清理

### 移除的测试代码

从以下文件底部移除 `if __name__ == "__main__":` 测试代码：

| 文件 | 移除行数 |
|------|----------|
| `src/data_modules.py` | ~30 |
| `src/model_modules.py` | ~35 |
| `src/correction_modules.py` | ~35 |
| `src/uncertainty_modules.py` | ~40 |
| `src/metrics.py` | ~25 |

### 移除的冗余函数

- `src/splitters.py`: `_ref_split_ratio()` - 未使用

### 清理的导入

- `src/splitters.py`: 移除 `Tuple`, `Iterable`
- `src/interfaces.py`: 移除 `List`, `Union`
- `config.py`: 移除 `os`
- `src/logger.py`: 移除 `os`

---

## 五、接口增强

### DataModule 类文档

**修改前**：
```python
class DataModule(ABC):
    """
    M1 数据模块抽象基类
    
    职责：数据标准化、分布处理、输出样本权重
    约束：fit_transform() 仅训练折调用，transform() 仅验证折调用
    """
```

**修改后**：
```python
class DataModule(ABC):
    """
    M1 数据模块抽象基类
    
    职责：
    - 数据标准化（Z-score normalization）
    - 分布处理（平衡/增强）
    - 输出样本权重
    
    约束：
    - fit_transform() 仅在训练折调用
    - transform() 仅在验证/测试折调用
    - 所有拟合参数通过 DataModuleState 传递
    
    数据契约：
    - 输入 X: shape (n_samples, n_features), dtype=float64
      - 单位: wt%（氧化物质量百分比）
      - 范围: 通常 0-100，但可能有负值（测量误差）
    - 输入 y: shape (n_samples,)
      - 温度 T: 单位 °C，范围约 700-1500
      - 压力 P: 单位 kbar，范围约 0-25
    - 输出 X: 标准化后，均值≈0，标准差≈1
    - 输出 sample_weights: 非负，总和≈n_samples
    """
```

---

### 结果版本管理

`protocol.py` 的 `save_config()` 方法现在自动添加版本信息：

```yaml
# config_used.yaml 新增内容
version_info:
  python_version: "3.11.5 ..."
  platform: "win32"
  timestamp: "2026-02-01T10:30:00"
  git_commit: "abc123..."
  git_dirty: false
  dependencies:
    numpy: "1.26.4"
    pandas: "2.2.0"
    scikit-learn: "1.4.0"
    catboost: "1.2.7"
```

---

## 六、文件变更汇总

### 新增文件（9个）

```
config.py                           # 集中配置管理
src/logger.py                       # 统一日志模块
tests/__init__.py                   # 测试包（已存在）
tests/conftest.py                   # pytest fixtures
tests/test_data_modules.py          # M1 测试
tests/test_model_modules.py         # M2 测试
tests/test_correction_modules.py    # M3 测试
tests/test_protocol.py              # 协议测试
tests/test_splitters.py             # 划分工具测试
tests/test_metrics.py               # 指标测试
```

### 修改文件（13个）

```
main.py                             # 脚本名称引用修正
README.md                           # V6 变更记录、项目结构更新、乱码修复
requirements.txt                    # 添加 pytest
src/__init__.py                     # 导出 logger
src/interfaces.py                   # 文档增强、TYPE_CHECKING
src/data_modules.py                 # 移除测试代码、修复 M2
src/model_modules.py                # 移除测试代码
src/correction_modules.py           # 移除测试代码
src/uncertainty_modules.py          # 移除测试代码
src/metrics.py                      # 移除测试代码
src/protocol.py                     # 日志、版本管理、修复 m1/M4
src/splitters.py                    # 移除冗余函数和导入
src/viz.py                          # 导入保护
tools/run_learning_curve.py         # 乱码修复
```

---

## 七、向后兼容性

### 完全兼容

- 现有 `main.py` 和 `tools/*.py` 无需修改即可运行
- `get_legacy_config()` 提供与旧 `CONFIG` 字典相同的接口

### 推荐迁移

旧代码：
```python
from main import CONFIG
data_path = CONFIG['data_path']
```

新代码：
```python
from config import CONFIG
data_path = CONFIG.data.path
```

---

## 八、验证结果

```
$ pytest tests/ -v
============================= 87 passed in 2.21s ==============================
```

所有 87 个测试用例通过，包括：
- 数据模块：16 passed
- 模型模块：18 passed
- 校正模块：12 passed
- 协议：9 passed
- 划分工具：15 passed
- 指标：18 passed

---

## 九、后续建议

1. **生成 requirements-lock.txt**：`pip freeze > requirements-lock.txt`
2. **配置 CI/CD**：添加 GitHub Actions 运行测试
3. **增加测试覆盖率目标**：建议 ≥80%
4. **日志轮转**：生产环境建议添加日志轮转配置
