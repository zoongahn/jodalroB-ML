import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import seaborn as sns

# 모델 로드
# model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")
model = SentenceTransformer("BAAI/bge-m3")

sentence = "2012년 양천권역 임대아파트 시설물 보수공사"

print("=" * 60)
print("1. 모델 정보 확인")
print("=" * 60)

# 모델 구조 확인
print(f"모델 구조: {model}")
print(f"최대 시퀀스 길이: {model.max_seq_length}")
print(f"임베딩 차원: {model.get_sentence_embedding_dimension()}")

# 토크나이저 정보
tokenizer = model.tokenizer
print(f"토크나이저 타입: {type(tokenizer)}")
print(f"어휘 크기: {tokenizer.vocab_size}")

print("\n" + "=" * 60)
print("2. 토큰화 과정 분석")
print("=" * 60)

# 토큰화 과정 상세 분석
tokens = tokenizer.tokenize(sentence)
token_ids = tokenizer.encode(sentence, add_special_tokens=True)
decoded_tokens = [tokenizer.decode([token_id]) for token_id in token_ids]

print(f"원본 문장: {sentence}")
print(f"토큰 개수: {len(tokens)}")
print(f"토큰들: {tokens}")
print(f"토큰 ID들: {token_ids}")
print(f"디코딩된 토큰들: {decoded_tokens}")

# 특수 토큰 확인
special_tokens = {
	'CLS': tokenizer.cls_token_id,
	'SEP': tokenizer.sep_token_id,
	'PAD': tokenizer.pad_token_id,
	'UNK': tokenizer.unk_token_id
}
print(f"특수 토큰 ID들: {special_tokens}")

print("\n" + "=" * 60)
print("3. 임베딩 생성 과정")
print("=" * 60)

# 임베딩 생성
embeddings = model.encode([sentence], convert_to_tensor=True, show_progress_bar=False)
embedding_vector = embeddings[0]

print(f"임베딩 shape: {embedding_vector.shape}")
print(f"임베딩 타입: {type(embedding_vector)}")
print(f"임베딩 dtype: {embedding_vector.dtype}")

# 임베딩 통계
print(f"임베딩 최소값: {embedding_vector.min():.6f}")
print(f"임베딩 최대값: {embedding_vector.max():.6f}")
print(f"임베딩 평균: {embedding_vector.mean():.6f}")
print(f"임베딩 표준편차: {embedding_vector.std():.6f}")
print(f"임베딩 L2 norm: {torch.norm(embedding_vector).item():.6f}")

print("\n" + "=" * 60)
print("4. 세부 레이어별 분석")
print("=" * 60)

# 모델의 각 모듈 확인
for i, module in enumerate(model.modules()):
	print(f"모듈 {i}: {type(module).__name__}")


# Transformer 모델의 hidden states 추출
def get_hidden_states(model, sentence):
	"""각 레이어의 hidden states 추출"""
	model.eval()

	# 입력 토큰화
	encoded = model.tokenizer(sentence, return_tensors='pt', padding=True, truncation=True)

	# 첫 번째 모듈이 transformer 모델
	transformer_model = model[0].auto_model

	# hidden states와 attention 가져오기
	with torch.no_grad():
		outputs = transformer_model(**encoded, output_hidden_states=True, output_attentions=True)

	return {
		'hidden_states': outputs.hidden_states,  # 각 레이어의 출력
		'attentions': outputs.attentions,  # 어텐션 가중치
		'last_hidden_state': outputs.last_hidden_state,
		'input_ids': encoded['input_ids']
	}


# Hidden states 분석
hidden_info = get_hidden_states(model, sentence)

print(f"레이어 수: {len(hidden_info['hidden_states'])}")
print(f"각 레이어 출력 shape: {hidden_info['hidden_states'][0].shape}")
print(f"어텐션 헤드 수: {len(hidden_info['attentions'])}")

# 각 레이어별 변화 분석
layer_norms = []
for i, hidden_state in enumerate(hidden_info['hidden_states']):
	# CLS 토큰의 representation 사용 (보통 첫 번째 토큰)
	cls_representation = hidden_state[0, 0, :]  # [batch, seq, hidden] -> [hidden]
	norm = torch.norm(cls_representation).item()
	layer_norms.append(norm)
	print(f"레이어 {i} CLS 토큰 norm: {norm:.4f}")

print("\n" + "=" * 60)
print("5. 어텐션 패턴 분석")
print("=" * 60)

# 마지막 레이어의 어텐션 분석
last_attention = hidden_info['attentions'][-1]  # 마지막 레이어
print(f"어텐션 shape: {last_attention.shape}")  # [batch, heads, seq_len, seq_len]

# 평균 어텐션 계산 (모든 헤드의 평균)
avg_attention = last_attention[0].mean(dim=0)  # 헤드 차원에서 평균
print(f"평균 어텐션 shape: {avg_attention.shape}")

# 각 토큰에 대한 어텐션 가중치
token_attention_weights = avg_attention[0]  # CLS 토큰이 다른 토큰들에 주는 어텐션
print("CLS 토큰의 각 토큰에 대한 어텐션 가중치:")
for i, (token, weight) in enumerate(zip(decoded_tokens, token_attention_weights)):
	print(f"  {i:2d}: {token:10s} -> {weight:.4f}")

print("\n" + "=" * 60)
print("6. 임베딩 차원별 분석")
print("=" * 60)

# 임베딩의 각 차원 분석
embedding_np = embedding_vector.cpu().numpy()

# 가장 큰/작은 값을 가진 차원들
top_indices = np.argsort(np.abs(embedding_np))[-10:]
print("절댓값이 가장 큰 10개 차원:")
for idx in reversed(top_indices):
	print(f"  차원 {idx:3d}: {embedding_np[idx]:8.4f}")

print(f"\n0에 가까운 값들의 비율: {(np.abs(embedding_np) < 0.1).mean():.2%}")
print(f"양수 값들의 비율: {(embedding_np > 0).mean():.2%}")

print("\n" + "=" * 60)
print("7. 임베딩 시각화 준비")
print("=" * 60)


def visualize_embeddings():
	"""임베딩 시각화"""
	fig, axes = plt.subplots(2, 2, figsize=(15, 10))

	# 1. 임베딩 벡터 히스토그램
	axes[0, 0].hist(embedding_np, bins=50, alpha=0.7)
	axes[0, 0].set_title('임베딩 값 분포')
	axes[0, 0].set_xlabel('값')
	axes[0, 0].set_ylabel('빈도')

	# 2. 임베딩 벡터 라인 플롯 (처음 100개 차원)
	axes[0, 1].plot(embedding_np[:100])
	axes[0, 1].set_title('임베딩 벡터 (처음 100차원)')
	axes[0, 1].set_xlabel('차원')
	axes[0, 1].set_ylabel('값')

	# 3. 레이어별 norm 변화
	axes[1, 0].plot(layer_norms, marker='o')
	axes[1, 0].set_title('레이어별 CLS 토큰 Norm')
	axes[1, 0].set_xlabel('레이어')
	axes[1, 0].set_ylabel('L2 Norm')

	# 4. 어텐션 히트맵
	sns.heatmap(avg_attention.cpu().numpy(),
	            xticklabels=decoded_tokens,
	            yticklabels=decoded_tokens,
	            ax=axes[1, 1], cmap='Blues')
	axes[1, 1].set_title('어텐션 가중치')
	axes[1, 1].tick_params(axis='x', rotation=45)

	plt.tight_layout()
	plt.show()


print("시각화 함수가 준비되었습니다. visualize_embeddings() 함수를 호출하여 그래프를 확인하세요.")

print("\n" + "=" * 60)
print("8. 풀링 전략 확인")
print("=" * 60)

# SentenceTransformer가 사용하는 풀링 전략 확인
pooling_layer = None
for module in model.modules():
	if hasattr(module, 'pooling_mode_mean_tokens'):
		pooling_layer = module
		break

if pooling_layer:
	print("풀링 전략:")
	print(f"  Mean tokens: {pooling_layer.pooling_mode_mean_tokens}")
	print(f"  CLS token: {pooling_layer.pooling_mode_cls_token}")
	print(f"  Max tokens: {pooling_layer.pooling_mode_max_tokens}")
else:
	print("풀링 레이어를 찾을 수 없습니다.")

print("\n" + "=" * 60)
print("9. 정규화 확인")
print("=" * 60)

# 정규화 여부 확인
raw_embeddings = model.encode([sentence], normalize_embeddings=False)
normalized_embeddings = model.encode([sentence], normalize_embeddings=True)

print(f"정규화 전 L2 norm: {np.linalg.norm(raw_embeddings[0]):.6f}")
print(f"정규화 후 L2 norm: {np.linalg.norm(normalized_embeddings[0]):.6f}")

# 함수들을 호출하여 분석 실행
print("\n분석 완료! 위의 정보들을 통해 임베딩의 내부 구조를 파악할 수 있습니다.")
