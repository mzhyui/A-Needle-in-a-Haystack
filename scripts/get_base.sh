#!/bin/bash

shared_args=""
while getopts "f:s:" opt; do
  case $opt in
    f)
        filepth="$OPTARG"
        ;;
    s)
        shared_args="$OPTARG"
        ;;
    *)
      echo "Usage: $0 [-f sh file path]" >&2
      exit 1
      ;;
  esac
done

# wrapper_script.sh
TEMP_FILE=$(mktemp)
BASEDIR_FILE=$(mktemp)

# # 运行原始脚本，同时捕获输出和过滤basedir行
# bash "$filepth" "$shared_args" 2>&1 | tee "$TEMP_FILE" | while IFS= read -r line; do
#     echo "$line"
#     if [[ "$line" == *"Base_dir:"* ]]; then
#         echo "$line" >> "$BASEDIR_FILE"
#     fi
# done

# 逐行读取并执行原始脚本中的命令（更安全的方式）
while IFS= read -r script_line || [[ -n "$script_line" ]]; do
    # 跳过空行和注释行
    if [[ -z "$script_line" ]] || [[ "$script_line" =~ ^[[:space:]]*# ]]; then
        continue
    fi
    
    # 跳过shebang行
    if [[ "$script_line" =~ ^#! ]]; then
        continue
    fi
    
    echo "running: $script_line $shared_args"
    
    # 创建临时脚本文件来执行单行命令
    temp_script=$(mktemp)
    echo "#!/bin/bash" > "$temp_script"
    echo "$script_line $shared_args" >> "$temp_script"
    chmod +x "$temp_script"
    
    # 执行临时脚本并捕获输出
    bash "$temp_script" 2>&1 | while IFS= read -r output_line; do
        echo "$output_line"
        echo "$output_line" >> "$TEMP_FILE"
        
        # 检查并保存basedir行
        if [[ "$output_line" == *"Base_dir:"* ]]; then
            echo "$output_line" >> "$BASEDIR_FILE"
        fi
    done
    
    # 清理临时文件
    rm -f "$temp_script"
    
done < "$filepth"

echo ""
echo "=================================================="
echo "'basedir' lines:"
echo "=================================================="

if [[ -s "$BASEDIR_FILE" ]]; then
    cat "$BASEDIR_FILE"
else
    echo "No 'basedir' found"
fi

# 清理临时文件
rm -f "$TEMP_FILE" "$BASEDIR_FILE"