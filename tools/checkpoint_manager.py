# -*- coding: utf-8 -*-
"""
长时间实验的断点续跑管理器

提供统一的分段运行、检查点保存、断点续跑、结果合并功能。
适用于学习曲线和稳定性测试等长时间运行的实验。

使用方式：
    from tools.checkpoint_manager import CheckpointManager

    # 创建管理器
    manager = CheckpointManager(
        output_dir="results/learning_curve",
        exp_name="lc_ert_stacking",
        checkpoint_interval=10
    )

    # 获取待运行的任务
    pending = manager.get_pending_tasks(total_tasks=100)

    # 运行并保存
    for task_id in pending:
        result = run_task(task_id)
        manager.save_result(task_id, result)

    # 合并结果
    final_df = manager.merge_results()
"""

import os
import glob
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime

import pandas as pd


class CheckpointManager:
    """
    检查点管理器 - 统一管理长时间实验的断点续跑

    功能：
    - 自动检测已完成的任务
    - 定期保存检查点
    - 支持断点续跑
    - 合并分段结果

    文件结构：
        {output_dir}/
        ├── {exp_name}_progress.json      # 进度文件（已完成的任务ID）
        ├── {exp_name}_results.csv        # 增量结果文件
        ├── {exp_name}_checkpoint_N.csv   # 检查点文件
        └── {exp_name}_final.csv          # 最终合并结果
    """

    def __init__(
        self,
        output_dir: str,
        exp_name: str,
        checkpoint_interval: int = 50,
        auto_resume: bool = True,
    ):
        """
        Parameters
        ----------
        output_dir : str
            输出目录
        exp_name : str
            实验名称（用作文件前缀）
        checkpoint_interval : int
            检查点间隔（每 N 个任务保存一次）
        auto_resume : bool
            是否自动从上次中断处继续
        """
        self.output_dir = Path(output_dir)
        self.exp_name = exp_name
        self.checkpoint_interval = checkpoint_interval
        self.auto_resume = auto_resume

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 文件路径
        self.progress_file = self.output_dir / f"{exp_name}_progress.json"
        self.results_file = self.output_dir / f"{exp_name}_results.csv"
        self.final_file = self.output_dir / f"{exp_name}_final.csv"

        # 内存中的结果缓存
        self._results_buffer: List[Dict] = []
        self._completed_tasks: Set[int] = set()
        self._task_count = 0

        # 加载已有进度
        if auto_resume:
            self._load_progress()

    def _load_progress(self) -> None:
        """加载已有进度"""
        # 从进度文件加载
        if self.progress_file.exists():
            try:
                with open(self.progress_file, 'r') as f:
                    data = json.load(f)
                    self._completed_tasks = set(data.get('completed_tasks', []))
                    print(f"  [CheckpointManager] 已加载进度: {len(self._completed_tasks)} 个任务已完成")
            except Exception as e:
                print(f"  [CheckpointManager] 加载进度文件失败: {e}")

        # 从结果文件补充
        if self.results_file.exists():
            try:
                df = pd.read_csv(self.results_file)
                if 'task_id' in df.columns:
                    file_tasks = set(df['task_id'].dropna().astype(int).tolist())
                    self._completed_tasks.update(file_tasks)
            except Exception:
                pass

    def _save_progress(self) -> None:
        """保存进度"""
        data = {
            'completed_tasks': sorted(list(self._completed_tasks)),
            'last_updated': datetime.now().isoformat(),
            'total_completed': len(self._completed_tasks),
        }
        with open(self.progress_file, 'w') as f:
            json.dump(data, f, indent=2)

    def get_pending_tasks(self, total_tasks: int, task_ids: Optional[List[int]] = None) -> List[int]:
        """
        获取待运行的任务列表

        Parameters
        ----------
        total_tasks : int
            总任务数
        task_ids : List[int], optional
            指定的任务ID列表，None 则使用 0 到 total_tasks-1

        Returns
        -------
        List[int]
            待运行的任务ID列表
        """
        if task_ids is None:
            task_ids = list(range(total_tasks))

        pending = [t for t in task_ids if t not in self._completed_tasks]

        if self.auto_resume and len(pending) < len(task_ids):
            print(f"  [CheckpointManager] 跳过 {len(task_ids) - len(pending)} 个已完成任务")

        return pending

    def save_result(self, task_id: int, result: Dict[str, Any]) -> None:
        """
        保存单个任务结果

        Parameters
        ----------
        task_id : int
            任务ID
        result : Dict
            任务结果
        """
        result['task_id'] = task_id
        self._results_buffer.append(result)
        self._completed_tasks.add(task_id)
        self._task_count += 1

        # 定期保存检查点
        if self._task_count % self.checkpoint_interval == 0:
            self._flush_buffer()
            self._save_checkpoint()

    def _flush_buffer(self) -> None:
        """将缓冲区写入结果文件"""
        if not self._results_buffer:
            return

        df = pd.DataFrame(self._results_buffer)

        # 追加模式写入
        if self.results_file.exists():
            df.to_csv(self.results_file, mode='a', header=False, index=False)
        else:
            df.to_csv(self.results_file, index=False)

        self._results_buffer.clear()
        self._save_progress()

    def _save_checkpoint(self) -> None:
        """保存检查点"""
        checkpoint_file = self.output_dir / f"{self.exp_name}_checkpoint_{self._task_count}.csv"

        # 读取所有已保存的结果
        if self.results_file.exists():
            df = pd.read_csv(self.results_file)
            df.to_csv(checkpoint_file, index=False)
            print(f"  [Checkpoint] 已保存 {len(df)} 条记录 -> {checkpoint_file.name}")

    def finalize(self) -> pd.DataFrame:
        """
        完成实验，合并所有结果

        Returns
        -------
        pd.DataFrame
            最终合并的结果
        """
        # 刷新缓冲区
        self._flush_buffer()

        # 读取结果文件
        if not self.results_file.exists():
            print("  [CheckpointManager] 警告: 无结果文件")
            return pd.DataFrame()

        df = pd.read_csv(self.results_file)

        # 去重（按 task_id）
        if 'task_id' in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=['task_id'], keep='last')
            after = len(df)
            if before != after:
                print(f"  [CheckpointManager] 去除 {before - after} 条重复记录")

        # 保存最终文件
        df.to_csv(self.final_file, index=False)
        print(f"  [CheckpointManager] 最终结果: {len(df)} 条 -> {self.final_file.name}")

        # 清理检查点文件（可选）
        # self._cleanup_checkpoints()

        return df

    def _cleanup_checkpoints(self) -> None:
        """清理检查点文件"""
        pattern = str(self.output_dir / f"{self.exp_name}_checkpoint_*.csv")
        for f in glob.glob(pattern):
            os.remove(f)

    @classmethod
    def merge_segments(
        cls,
        segment_dirs: List[str],
        output_dir: str,
        exp_name: str,
        task_id_col: str = 'task_id',
    ) -> pd.DataFrame:
        """
        合并多个分段运行的结果

        Parameters
        ----------
        segment_dirs : List[str]
            分段目录列表
        output_dir : str
            输出目录
        exp_name : str
            实验名称
        task_id_col : str
            任务ID列名

        Returns
        -------
        pd.DataFrame
            合并后的结果
        """
        all_dfs = []

        for seg_dir in segment_dirs:
            seg_path = Path(seg_dir)

            # 查找结果文件
            results_file = seg_path / f"{exp_name}_results.csv"
            final_file = seg_path / f"{exp_name}_final.csv"

            if final_file.exists():
                df = pd.read_csv(final_file)
            elif results_file.exists():
                df = pd.read_csv(results_file)
            else:
                # 查找任意 CSV 文件
                csvs = list(seg_path.glob(f"{exp_name}*.csv"))
                if csvs:
                    df = pd.read_csv(csvs[0])
                else:
                    print(f"  警告: {seg_dir} 中无结果文件")
                    continue

            all_dfs.append(df)

        if not all_dfs:
            raise ValueError("未找到任何结果文件")

        merged = pd.concat(all_dfs, ignore_index=True)

        # 去重
        if task_id_col in merged.columns:
            before = len(merged)
            merged = merged.drop_duplicates(subset=[task_id_col], keep='last')
            after = len(merged)
            if before != after:
                print(f"  去除 {before - after} 条重复记录")

        # 保存
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        final_path = output_path / f"{exp_name}_final.csv"
        merged.to_csv(final_path, index=False)
        print(f"  合并结果: {len(merged)} 条 -> {final_path}")

        return merged


def get_segment_ranges(total: int, segment_size: int) -> List[tuple]:
    """
    计算分段范围

    Parameters
    ----------
    total : int
        总任务数
    segment_size : int
        每段大小

    Returns
    -------
    List[tuple]
        [(start, end), ...] 列表
    """
    ranges = []
    for start in range(0, total, segment_size):
        end = min(start + segment_size - 1, total - 1)
        ranges.append((start, end))
    return ranges


def print_segment_commands(
    script_name: str,
    total: int,
    segment_size: int,
    extra_args: str = "",
) -> None:
    """
    打印分段运行命令（便于复制）

    Parameters
    ----------
    script_name : str
        脚本名称
    total : int
        总任务数
    segment_size : int
        每段大小
    extra_args : str
        额外参数
    """
    ranges = get_segment_ranges(total, segment_size)

    print(f"\n{'='*60}")
    print(f"分段运行命令（共 {len(ranges)} 段）")
    print(f"{'='*60}")

    for i, (start, end) in enumerate(ranges):
        cmd = f"python {script_name} --repeat-start {start} --repeat-end {end} {extra_args}"
        print(f"# 段 {i+1}/{len(ranges)}")
        print(cmd)
        print()

    print(f"# 合并结果")
    print(f"python {script_name} --merge-dir results {extra_args}")
