import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import pandas as pd
from datetime import datetime
import time
import multiprocessing
from concurrent.futures import ThreadPoolExecutor

class TrafficQuerySystem:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Inquiry System For Taiwan Traffic Data")
        self.window.geometry("1400x800")
        
        # 定义列顺序
        self.columns_order = [
            'VehicleType',
            'DerectionTime_O',
            'GantryID_O',
            'DerectionTime_D',
            'GantryID_D',
            'TripLength',
            'TripEnd',
            'TripInformation'
        ]
        
        # 列宽度设置
        self.column_widths = {
            'VehicleType': 80,
            'DerectionTime_O': 150,
            'GantryID_O': 100,
            'DerectionTime_D': 150,
            'GantryID_D': 100,
            'TripLength': 80,
            'TripEnd': 60,
            'TripInformation': 400
        }
        
        self.df = None
        self.create_widgets()

    def optimized_counting_sort(self, arr, key_func=None):
        """优化的计数排序实现"""
        if not arr:
            return arr
        
        try:
            # 使用key_func获取所有值并转换为整数
            if key_func:
                values = [int(float(key_func(x))) for x in arr]
            else:
                values = [int(float(x)) for x in arr]
            
            if not values:
                return arr
                
            min_val = min(values)
            max_val = max(values)
            range_size = max_val - min_val + 1
            
            # 如果范围太大，切换到快速排序
            if range_size > len(arr) * 2:
                return self.optimized_quick_sort(arr, key_func)
            
            # 创建计数数组和值到原始项的映射
            count = [0] * range_size
            value_to_items = {}
            
            # 计数并建立映射
            for i, val in enumerate(values):
                count[val - min_val] += 1
                if val not in value_to_items:
                    value_to_items[val] = []
                value_to_items[val].append(arr[i])
            
            # 重建排序后的数组
            sorted_arr = []
            for i in range(range_size):
                val = i + min_val
                if count[i] > 0 and val in value_to_items:
                    sorted_arr.extend(value_to_items[val])
            
            return sorted_arr
            
        except (ValueError, TypeError) as e:
            print(f"Counting sort failed: {str(e)}, falling back to quick sort")
            return self.optimized_quick_sort(arr, key_func)

    def optimized_merge_sort(self, arr, key_func=None):
        """优化的归并排序实现"""
        if len(arr) <= 1:
            return arr
            
        try:
            mid = len(arr) // 2
            left = arr[:mid]
            right = arr[mid:]
            
            # 递归排序
            left = self.optimized_merge_sort(left, key_func)
            right = self.optimized_merge_sort(right, key_func)
            
            # 合并
            return self._merge(left, right, key_func)
            
        except Exception as e:
            print(f"Merge sort failed: {str(e)}, falling back to quick sort")
            return self.optimized_quick_sort(arr, key_func)

    def _merge(self, left, right, key_func=None):
        """合并两个有序数组"""
        result = []
        i = j = 0
        
        try:
            while i < len(left) and j < len(right):
                # 获取比较值
                left_val = key_func(left[i]) if key_func else left[i]
                right_val = key_func(right[j]) if key_func else right[j]
                
                # 处理None值
                if left_val is None:
                    left_val = float('-inf')
                if right_val is None:
                    right_val = float('-inf')
                
                # 比较并添加元素
                if left_val <= right_val:
                    result.append(left[i])
                    i += 1
                else:
                    result.append(right[j])
                    j += 1
            
            # 添加剩余元素
            result.extend(left[i:])
            result.extend(right[j:])
            return result
            
        except Exception as e:
            print(f"Merge operation failed: {str(e)}")
            # 在合并失败时返回连接的列表
            return left + right

    def optimized_quick_sort(self, arr, key_func=None):
        """优化的快速排序实现"""
        if len(arr) <= 1:
            return arr
            
        try:
            # 选择pivot（使用中间元素）
            pivot = arr[len(arr) // 2]
            pivot_val = key_func(pivot) if key_func else pivot
            
            # 处理None值
            if pivot_val is None:
                pivot_val = float('-inf')
            
            # 分区
            left = []
            middle = []
            right = []
            
            for x in arr:
                val = key_func(x) if key_func else x
                if val is None:
                    val = float('-inf')
                    
                if val < pivot_val:
                    left.append(x)
                elif val == pivot_val:
                    middle.append(x)
                else:
                    right.append(x)
            
            # 递归排序并组合结果
            return (self.optimized_quick_sort(left, key_func) + 
                    middle + 
                    self.optimized_quick_sort(right, key_func))
                    
        except Exception as e:
            print(f"Quick sort failed: {str(e)}")
            # 如果排序失败，返回原始数组
            return arr

    def optimized_heap_sort(self, arr, key_func=None):
        """优化的堆排序实现"""
        def heapify(arr, n, i):
            try:
                largest = i
                left = 2 * i + 1
                right = 2 * i + 2
                
                # 获取和比较值
                def get_val(idx):
                    if idx < n:
                        val = key_func(arr[idx]) if key_func else arr[idx]
                        return float('-inf') if val is None else val
                    return float('-inf')
                
                largest_val = get_val(largest)
                left_val = get_val(left)
                right_val = get_val(right)
                
                # 更新最大值的索引
                if left < n and left_val > largest_val:
                    largest = left
                    largest_val = left_val
                
                if right < n and right_val > largest_val:
                    largest = right
                
                # 如果最大值不是根节点，则交换并继续堆化
                if largest != i:
                    arr[i], arr[largest] = arr[largest], arr[i]
                    heapify(arr, n, largest)
                    
            except Exception as e:
                print(f"Heapify operation failed: {str(e)}")
                return
        
        try:
            # 创建数组副本
            arr = arr.copy()
            n = len(arr)
            
            # 构建最大堆
            for i in range(n // 2 - 1, -1, -1):
                heapify(arr, n, i)
            
            # 一个个提取元素
            for i in range(n - 1, 0, -1):
                arr[0], arr[i] = arr[i], arr[0]
                heapify(arr, i, 0)
            
            return arr
            
        except Exception as e:
            print(f"Heap sort failed: {str(e)}, falling back to quick sort")
            return self.optimized_quick_sort(arr, key_func)
    
    def parallel_sort(self, arr, sort_function, key_func=None):
        """并行排序实现"""
        try:
            # 对于小数据集直接排序
            if len(arr) < 1000:
                return sort_function(arr, key_func)

            # 确定分块数量
            num_chunks = min(len(arr) // 1000, multiprocessing.cpu_count() * 2)
            if num_chunks < 2:
                return sort_function(arr, key_func)

            # 分割数据
            chunk_size = len(arr) // num_chunks
            chunks = [arr[i:i + chunk_size] for i in range(0, len(arr), chunk_size)]

            # 并行排序
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                sorted_chunks = list(executor.map(
                    lambda x: sort_function(x, key_func), 
                    chunks
                ))

            # 合并排序后的块
            return self._merge_sorted_arrays(sorted_chunks, key_func)
            
        except Exception as e:
            print(f"Parallel sort failed: {str(e)}, falling back to sequential sort")
            return sort_function(arr, key_func)

    def _merge_sorted_arrays(self, arrays, key_func=None):
        """合并多个有序数组"""
        if not arrays:
            return []
        if len(arrays) == 1:
            return arrays[0]

        # 使用分治法合并
        mid = len(arrays) // 2
        left = self._merge_sorted_arrays(arrays[:mid], key_func)
        right = self._merge_sorted_arrays(arrays[mid:], key_func)
        return self._merge(left, right, key_func)

    def create_widgets(self):
        # 创建主框架
        self.main_frame = ttk.Frame(self.window, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # 创建标题
        title_label = ttk.Label(self.main_frame, 
                              text="Inquiry System For Taiwan Traffic Data",
                              font=('Arial', 16, 'bold'))
        title_label.pack(pady=10)
        
        # 创建左右分栏框架
        content_frame = ttk.Frame(self.main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # 左侧控制面板
        left_frame = ttk.Frame(content_frame, width=400)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5)
        
        # 右侧数据显示区域
        right_frame = ttk.Frame(content_frame)
        right_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        
        # 创建左侧控制元素
        self.create_control_panel(left_frame)
        
        # 创建右侧标签页
        self.create_data_tabs(right_frame)
        
        # 创建进度条框架
        self.create_progress_frame()
        
    def create_progress_frame(self):
        progress_frame = ttk.Frame(self.main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress = ttk.Progressbar(
            progress_frame,
            orient="horizontal",
            length=300,
            mode="determinate"
        )
        self.progress.pack(side=tk.LEFT, padx=5)
        
        # 分开显示操作信息和记录数量
        status_frame = ttk.Frame(progress_frame)
        status_frame.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.progress_label = ttk.Label(status_frame, text="Ready")
        self.progress_label.pack(side=tk.LEFT, padx=5)
        
        self.record_count_label = ttk.Label(status_frame, text="")
        self.record_count_label.pack(side=tk.LEFT, padx=5)
        
    def create_control_panel(self, parent):
        # Import/Export按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(btn_frame, text="import", command=self.import_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="export", command=self.export_data).pack(side=tk.LEFT, padx=5)
        
        # Search框架
        search_frame = ttk.LabelFrame(parent, text="Search", padding=5)
        search_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(search_frame, text="search records(column)").pack(anchor=tk.W)
        self.search_column = ttk.Combobox(search_frame, values=self.columns_order)
        self.search_column.pack(fill=tk.X, pady=2)
        
        ttk.Label(search_frame, text="keyword").pack(anchor=tk.W)
        self.search_keyword = ttk.Entry(search_frame)
        self.search_keyword.pack(fill=tk.X, pady=2)
        
        ttk.Button(search_frame, text="search button", command=self.perform_search).pack(pady=5)
        
        # Sort框架
        sort_frame = ttk.LabelFrame(parent, text="Sort", padding=5)
        sort_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(sort_frame, text="sorts records(column)").pack(anchor=tk.W)
        self.sort_column = ttk.Combobox(sort_frame, values=self.columns_order)
        self.sort_column.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_frame, text="Ascending order or not").pack(anchor=tk.W)
        self.sort_order = ttk.Combobox(sort_frame, values=["ascending", "descending"])
        self.sort_order.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_frame, text="Max number of items").pack(anchor=tk.W)
        self.sort_limit = ttk.Entry(sort_frame)
        self.sort_limit.insert(0, "default")
        self.sort_limit.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_frame, text="choose the sort algorithm").pack(anchor=tk.W)
        self.sort_algorithm = ttk.Combobox(sort_frame, values=[
            "Quick sort",
            "Merge sort", 
            "Heap sort",
            "Counting sort"
        ])
        self.sort_algorithm.pack(fill=tk.X, pady=2)
        
        ttk.Label(sort_frame, text="choose the subset of the data").pack(anchor=tk.W)
        self.sort_subset = ttk.Combobox(sort_frame, values=["100%", "50%", "33%", "25%", "20%"])
        self.sort_subset.pack(fill=tk.X, pady=2)
        
        ttk.Button(sort_frame, text="sort button", command=self.perform_sort).pack(pady=5)
        
        # Join框架
        join_frame = ttk.LabelFrame(parent, text="Join", padding=5)
        join_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(join_frame, text="join records(column)").pack(anchor=tk.W)
        self.join_column = ttk.Combobox(join_frame, values=self.columns_order)
        self.join_column.pack(fill=tk.X, pady=2)
        
        ttk.Button(join_frame, text="join button", command=self.perform_join).pack(pady=5)

    def create_data_tabs(self, parent):
        # 创建标签页
        self.notebook = ttk.Notebook(parent)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # 原始数据标签页
        self.original_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.original_frame, text="Pre-operation data")
        
        # 处理后数据标签页
        self.processed_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.processed_frame, text="Post-operation data")
        
        # 在两个标签页中创建表格
        self.original_tree = self.create_treeview(self.original_frame)
        self.processed_tree = self.create_treeview(self.processed_frame)

    def create_treeview(self, parent):
        # 创建表格并设置列
        tree = ttk.Treeview(parent, columns=self.columns_order, show='headings', height=20)
        
        # 设置每列的表头和宽度
        for col in self.columns_order:
            tree.heading(col, text=col, anchor='center')
            width = self.column_widths.get(col, 100)
            tree.column(col, width=width, anchor='center')
        
        # 添加滚动条
        vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # 布局
        tree.grid(column=0, row=0, sticky='nsew')
        vsb.grid(column=1, row=0, sticky='ns')
        hsb.grid(column=0, row=1, sticky='ew')
        
        parent.grid_columnconfigure(0, weight=1)
        parent.grid_rowconfigure(0, weight=1)
        
        return tree

    def get_sort_key(self, column):
        """获取排序键函数"""
        def time_key(x):
            """时间字段的排序键"""
            try:
                time_str = str(x[column])
                if not time_str:
                    return datetime.min
                if '/' in time_str:
                    return datetime.strptime(time_str, '%Y/%m/%d %H:%M:%S')
                else:
                    return datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
            except:
                return datetime.min

        def number_key(x):
            """数值字段的排序键"""
            try:
                val = x[column]
                return float(val) if val else float('-inf')
            except:
                return float('-inf')

        def string_key(x):
            """字符串字段的排序键"""
            try:
                return str(x[column])
            except:
                return ''

        # 根据列类型返回相应的排序键函数
        if column in ['DerectionTime_O', 'DerectionTime_D']:
            return time_key
        elif column in ['TripLength', 'VehicleType']:
            return number_key
        else:
            return string_key

    def perform_sort(self):
        if self.df is None:
            messagebox.showerror("Error", "Please import data first!")
            return
            
        try:
            # 获取排序参数
            column = self.sort_column.get()
            ascending = self.sort_order.get() == "ascending"
            max_items = self.sort_limit.get()
            subset = self.sort_subset.get()
            algorithm = self.sort_algorithm.get()
            
            if not column or not algorithm:
                messagebox.showerror("Error", "Please select column and sort algorithm")
                return

            # 获取要排序的数据子集
            total_records = len(self.df)
            if subset != "100%":
                n = int(total_records * float(subset.strip('%')) / 100)
                df_to_sort = self.df.head(n)  # 从开始选取n条记录
            else:
                df_to_sort = self.df.copy()

            subset_records = len(df_to_sort)
            print(f"Sorting {subset_records} records ({subset} of {total_records} total records)")

            # 转换为记录列表
            records = df_to_sort.to_dict('records')
            key_func = self.get_sort_key(column)
            
            # 选择排序算法并计时
            start_time = time.time()
            if algorithm == "Counting sort" and column in ['VehicleType', 'TripLength']:
                sorted_records = self.parallel_sort(records, self.optimized_counting_sort, key_func)
            elif algorithm == "Merge sort":
                sorted_records = self.parallel_sort(records, self.optimized_merge_sort, key_func)
            elif algorithm == "Quick sort":
                sorted_records = self.parallel_sort(records, self.optimized_quick_sort, key_func)
            elif algorithm == "Heap sort":
                sorted_records = self.parallel_sort(records, self.optimized_heap_sort, key_func)
            else:
                raise ValueError("Invalid sorting algorithm selected")
            
            sort_time = time.time() - start_time

            # 转换回DataFrame并处理升序/降序
            sorted_df = pd.DataFrame(sorted_records)
            if not ascending:
                sorted_df = sorted_df.iloc[::-1].reset_index(drop=True)

            # 处理最大显示数量
            if max_items.lower() != 'default' and max_items.strip():
                try:
                    max_items = int(max_items)
                    sorted_df = sorted_df.head(max_items)
                except ValueError:
                    messagebox.showwarning("Warning", "Invalid max items value, showing all results")
            
            final_records = len(sorted_df)
            self.update_processed_treeview(sorted_df)
            
            # 更新状态显示
            status_text = (f"Sort completed in {sort_time:.4f} seconds | "
                         f"Processed {subset_records} records ({subset} of {total_records})")
            self.update_progress(100, status_text)
            self.record_count_label.config(text=f"Showing {final_records} records")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_progress(0, "Sort failed")
            self.record_count_label.config(text="")

    def update_progress(self, value, text):
        self.progress['value'] = value
        self.progress_label['text'] = text
        self.window.update_idletasks()

    def import_data(self):
        try:
            filenames = filedialog.askopenfilenames(
                filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
            )
            if filenames:
                self.update_progress(0, "Importing data...")
                if self.df is None:
                    self.df = pd.DataFrame(columns=self.columns_order)
                
                for i, filename in enumerate(filenames):
                    new_df = pd.read_csv(filename, header=None, names=self.columns_order, dtype=str)
                    self.df = pd.concat([self.df, new_df], ignore_index=True)
                    self.update_progress(((i + 1) / len(filenames)) * 100, 
                                      f"Imported file {i+1}/{len(filenames)}")
                
                total_records = len(self.df)
                self.update_original_treeview()
                self.update_progress(100, f"Import completed")
                self.record_count_label.config(text=f"Total records: {total_records}")
                
        except Exception as e:
            messagebox.showerror("Error", f"Import failed: {str(e)}")
            self.update_progress(0, "Import failed")
            self.record_count_label.config(text="")

    def update_original_treeview(self):
        self.update_treeview(self.original_tree, self.df)
        
    def update_processed_treeview(self, df):
        self.update_treeview(self.processed_tree, df)
        self.notebook.select(1)  # 切换到处理后数据标签页
        
    def update_treeview(self, tree, df):
        # 清空现有数据
        for item in tree.get_children():
            tree.delete(item)
            
        try:
            # 将DataFrame的每一行添加到treeview中
            for i in range(len(df)):
                values = df.iloc[i].tolist()
                tree.insert("", tk.END, values=values)
        except Exception as e:
            messagebox.showerror("Error", f"Error updating display: {str(e)}")

    def perform_search(self):
        if self.df is None:
            messagebox.showerror("Error", "Please import data first!")
            return
            
        try:
            start_time = time.time()
            column = self.search_column.get()
            keyword = self.search_keyword.get()
            
            if not column or not keyword:
                messagebox.showerror("Error", "Please select column and enter keyword")
                return

            total_records = len(self.df)
            # 根据不同列类型进行搜索
            if column in ['DerectionTime_O', 'DerectionTime_D']:
                # 支持多种时间格式搜索
                search_formats = [
                    keyword,  # 原始输入
                    keyword.replace('-', '/').replace(' 0', ' '),  # 2023/12/4 6:37:09
                    keyword.replace('/', '-').replace(' ', ' 0')   # 2023-12-04 06:37:09
                ]
                mask = self.df[column].apply(lambda x: any(fmt == x for fmt in search_formats))
                result_df = self.df[mask]
                
            elif column == 'TripInformation':
                # TripInformation列的精确搜索
                result_df = self.df[self.df[column] == keyword]
                if len(result_df) == 0:
                    # 如果精确匹配没找到，尝试部分匹配
                    result_df = self.df[self.df[column].str.contains(keyword, regex=False, na=False)]
                
            elif column == 'TripLength':
                try:
                    value = float(keyword)
                    result_df = self.df[self.df[column].astype(float) == value]
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid number")
                    return
            else:
                # 其他字段精确匹配
                result_df = self.df[self.df[column] == keyword]
            
            search_time = time.time() - start_time
            found_records = len(result_df)
            
            self.update_processed_treeview(result_df)
            status_text = f"Search completed in {search_time:.4f} seconds"
            self.update_progress(100, status_text)
            self.record_count_label.config(
                text=f"Found {found_records} of {total_records} records")
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.update_progress(0, "Search failed")
            self.record_count_label.config(text="")

    def perform_join(self):
        if self.df is None:
            messagebox.showerror("Error", "Please import data first!")
            return
            
        try:
            column = self.join_column.get()
            if not column:
                messagebox.showerror("Error", "Please select a column for join")
                return
                
            self.update_progress(30, "Processing data...")
            start_time = time.time()
            
            # 对选定列进行分组统计
            value_counts = self.df[column].value_counts()
            
            # 只选择出现次数大于1的值
            duplicate_values = value_counts[value_counts > 1].index.tolist()
            
            if not duplicate_values:
                messagebox.showinfo("Info", f"No duplicate values found in column {column}")
                self.update_progress(0, "Join produced no results")
                return
                
            # 只处理包含重复值的记录并按列排序
            result_df = self.df[self.df[column].isin(duplicate_values)].sort_values(column)
            
            join_time = time.time() - start_time
            
            if len(result_df) > 0:
                # 更新显示
                self.update_processed_treeview(result_df[self.columns_order])
                status_text = f"Join completed in {join_time:.4f} seconds"
                self.update_progress(100, status_text)
                
                # 更新统计信息
                total_records = len(self.df)
                duplicate_records = len(result_df)
                unique_groups = len(duplicate_values)
                
                self.record_count_label.config(
                    text=(f"Found {duplicate_records} records with duplicates "
                        f"in {unique_groups} groups "
                        f"(from {total_records} total records)")
                )
            else:
                messagebox.showinfo("Info", "No duplicate records found")
                self.update_progress(0, "Join produced no results")
                self.record_count_label.config(text="No duplicates found")
                
        except Exception as e:
            messagebox.showerror("Error", f"Join failed: {str(e)}")
            self.update_progress(0, "Join failed")
            self.record_count_label.config(text="")

        def update_processed_treeview(self, df):
            """更新处理后的数据视图"""
            # 清空现有数据
            for item in self.processed_tree.get_children():
                self.processed_tree.delete(item)
                
            try:
                # 检查DataFrame是否为空
                if df.empty:
                    return
                    
                # 更新树形视图的列
                current_columns = self.processed_tree['columns']
                new_columns = df.columns.tolist()
                
                if current_columns != new_columns:
                    # 重新配置列
                    self.processed_tree['columns'] = new_columns
                    for col in new_columns:
                        self.processed_tree.heading(col, text=col, anchor='center')
                        width = self.column_widths.get(col, 100)
                        self.processed_tree.column(col, width=width, anchor='center')
                
                # 添加数据
                for i in range(len(df)):
                    values = df.iloc[i].tolist()
                    self.processed_tree.insert("", tk.END, values=values)
                    
                # 切换到处理后数据标签页
                self.notebook.select(1)
                
            except Exception as e:
                messagebox.showerror("Error", f"Error updating display: {str(e)}")

    def export_data(self):
        if self.df is None:
            messagebox.showerror("Error", "No data to export!")
            return
            
        filename = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("Excel files", "*.xlsx")]
        )
        
        if filename:
            self.update_progress(0, "Exporting data...")
            try:
                start_time = time.time()
                
                current_tab = self.notebook.select()
                if current_tab == self.notebook.tabs()[0]:  # 原始数据标签页
                    df_to_export = self.df
                else:  # 处理后数据标签页
                    df_to_export = self.get_treeview_data(self.processed_tree)
                
                if filename.endswith('.csv'):
                    df_to_export.to_csv(filename, index=False)
                else:
                    df_to_export.to_excel(filename, index=False)
                    
                export_time = time.time() - start_time
                status_text = f"Export completed in {export_time:.4f} seconds"
                self.update_progress(100, status_text)
                self.record_count_label.config(text=f"Exported records: {len(df_to_export)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Export failed: {str(e)}")
                self.update_progress(0, "Export failed")
                self.record_count_label.config(text="")
    
    def get_treeview_data(self, tree):
        data = []
        for item in tree.get_children():
            data.append(tree.item(item)['values'])
        return pd.DataFrame(data, columns=self.columns_order)
        
    def run(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = TrafficQuerySystem()
    app.run()