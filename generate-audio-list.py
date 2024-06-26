# 入力ファイル名と出力ファイル名を指定します
input_filename = 'esd.list'
output_filename = 'filelist.txt'

# ファイルを読み込み、必要な部分を抽出して新しいファイルに書き込みます
with open(input_filename, 'r', encoding='utf-8') as infile, open(output_filename, 'w', encoding='utf-8') as outfile:
    for line in infile:
        # 各行を分割して、ファイル名とテキストを取得します
        parts = line.strip().split('|')
        if len(parts) == 4:
            file_name = parts[0]
            text = parts[3]
            # 新しい形式に変換して出力ファイルに書き込みます
            outfile.write(f"audio/{file_name}|{text}\n")

print(f"Extracted file names and texts have been saved to {output_filename}.")