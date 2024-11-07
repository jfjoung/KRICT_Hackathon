import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# CSV 파일 경로 설정 
file_path = '/workspaces/KRICT_Hackathon/model/linear/results/predictions.csv'  

# 데이터 불러오기
data = pd.read_csv(file_path)

# 'True Formation Energy'와 'Predicted Formation Energy' 열을 추출
true_values = data['True Formation Energy']
predicted_values = data['Predicted Formation Energy']

# R²와 RMSE 계산
r2 = r2_score(true_values, predicted_values)
rmse = np.sqrt(mean_squared_error(true_values, predicted_values))

# Parity plot 그리기
plt.figure(figsize=(8, 6))
plt.scatter(true_values, predicted_values, color='blue', label='Predictions')
plt.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], color='red', linestyle='--', label='Ideal (True = Predicted)')
plt.xlabel('True Formation Energy')
plt.ylabel('Predicted Formation Energy')
plt.title('Parity Plot')
plt.legend()

# R²와 RMSE 값을 텍스트로 추가
plt.text(0.1, 0.9, f'R²: {r2:.3f}', transform=plt.gca().transAxes, fontsize=12)
plt.text(0.1, 0.85, f'RMSE: {rmse:.3f}', transform=plt.gca().transAxes, fontsize=12)

# 그래프 출력
# plt.show()
# 그래프를 파일로 저장
plt.savefig('parity_plot.png')  # 저장할 파일 이름과 확장자 지정
plt.close()  # 그래프 닫기
