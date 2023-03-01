echo "Enter ID query fixed result:"
read query_id 
echo "Enter Port:"
read port 
python /workspace/competitions/AI_Challenge_2022/visualization/result.py \
--result_path "/workspace/competitions/AI_Challenge_2022/fixed_results/query-${query_id}.csv" \
--port ${port}