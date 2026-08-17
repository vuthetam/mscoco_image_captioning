import torch
from torch import Tensor, nn
from torch.nn import functional as F

from vocabulary import Vocabulary


def _length_normalized_score(score: float, length: int, alpha: float) -> float:
    # Giảm xu hướng ưu tiên caption quá ngắn của tổng log-probability.
    penalty = ((5 + length) / 6) ** alpha
    return score / penalty


@torch.inference_mode()
def generate_caption_beam_search(
    encoder: nn.Module,
    decoder: nn.Module,
    image: Tensor,
    vocab: Vocabulary,
    max_length: int,
    beam_size: int = 5,
    length_penalty: float = 0.7,
) -> str:
    """Sinh caption cho một ảnh bằng beam search."""
    if beam_size < 1:
        raise ValueError("beam_size must be at least 1")
    if max_length < 2:
        raise ValueError("max_length must be at least 2")

    # Chuẩn hóa ảnh về dạng batch [1, C, H, W].
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if image.ndim != 4 or image.size(0) != 1:
        raise ValueError("image must have shape [C, H, W] or [1, C, H, W]")

    device = next(encoder.parameters()).device
    memory: Tensor = encoder(image.to(device))

    # Beam ban đầu chỉ chứa <SOS> và có tổng log-probability bằng 0.
    sequences = torch.tensor([[vocab.sos_token_id]], device=device) # shape: [beam_count, current_length], ban đầu [1,1]
    scores = torch.zeros(1, device=device)                          # shape: scores = [beam_count], ban đầu là [1]
    completed: list[tuple[Tensor, float]] = []
    has_active_beams = True

    # Trừ một vị trí vì <SOS> đã có sẵn trong sequences.
    for _ in range(max_length - 1):
        # beam_count đóng vai trò như batch size
        beam_count = sequences.size(0)

        # [1, image_tokens, d_model] -> [beam_count, image_tokens, d_model].
        # expand tạo view broadcast: các beam cùng tham chiếu memory gốc, không copy dữ liệu ảnh.
        beam_memory = memory.expand(beam_count, -1, -1) 

        # padding_mask cho sequences
        padding_masks = torch.zeros_like(sequences, dtype=torch.bool) # [beam_count, current_lenght]

        # Chỉ lấy logits ở vị trí cuối để dự đoán token tiếp theo.
        logits = decoder(beam_memory, sequences, padding_masks)[:, -1, :] # [beam_count, current_lenght, vocab_size] -> [beam_count, vocab_size].
        # Chặn <PAD> và <SOS> trước khi chuẩn hóa để chúng không được chọn
        # và cũng không tham gia vào mẫu số của softmax.
        logits[:, vocab.pad_token_id] = -torch.inf
        logits[:, vocab.sos_token_id] = -torch.inf

        # Chuẩn hóa logits thành log-probability trên toàn bộ vocabulary.
        log_probs = F.log_softmax(logits, dim=-1)  # [beam_count, vocab_size]

        # Cộng điểm cũ của từng beam với log-probability của mọi token mới
        # Mỗi hàng tương ứng với một beam hiện tại
        # Mỗi phần tử là điểm của beam sau khi nối thêm token tại cột đó
        candidate_scores = scores.unsqueeze(1) + log_probs   # [beam_count, vocab_size]

        # Lấy dư ứng viên vì một số candidate có thể kết thúc bằng <EOS>.
        candidate_count = min(beam_size * 2, candidate_scores.numel())  # numel() trả về tổng số phần tử của tensor
        top_scores, top_indices = candidate_scores.flatten().topk(candidate_count)  # mặc định sắp xếp giảm dần

        # Sau khi flatten: chia nguyên tìm beam nguồn, lấy dư tìm token_id.
        # Tensor 1 chiều: [candidate_count]
        source_beams = top_indices // len(vocab)
        token_ids = top_indices % len(vocab)
 
        next_sequences: list[Tensor] = []  # lưu các sequence mới, mỗi phần tử là tensor 1 chiều
        next_scores: list[Tensor] = []     # lưu score mới, mỗi phần tử là tensor vô hướng

        for source_beam, token_id, score in zip(source_beams, token_ids, top_scores):
            # source_beam, token_id, score đều là tensor vô hướng (tensor 0 chiều).

            # Nối token mới vào đúng beam đã sinh ra candidate này.
            # sequences[source_beam] có shape [current_length] -> tensor 1 chiều
            # view(1) đổi tensor vô hướng thành tensor 1 chiều
            sequence = torch.cat([sequences[source_beam], token_id.view(1)])

            if token_id.item() == vocab.eos_token_id:
                completed.append((sequence, score.item()))
            elif len(next_sequences) < beam_size:
                next_sequences.append(sequence)
                next_scores.append(score)

            if len(next_sequences) == beam_size:
                break

        # Không còn active beam nghĩa là các candidate tốt nhất đều đã kết thúc.
        if not next_sequences:
            has_active_beams = False
            break

        # stack gom một list[Tensor] thành một tensor lớn hơn bằng cách thêm một chiều mới
        sequences = torch.stack(next_sequences)   
        scores = torch.stack(next_scores)         

        # Chỉ giữ completed beam tốt nhất để danh sách không tăng liên tục.
        completed = sorted(
            completed,
            key=lambda item: _length_normalized_score(
                item[1], item[0].size(0) - 1, length_penalty
            ),
            reverse=True,
        )[:beam_size]

    # Nếu đạt max_length trước <EOS>, active beam vẫn được xét làm kết quả.
    if has_active_beams:
        completed.extend(
            (sequence, score.item()) for sequence, score in zip(sequences, scores)
        )

    # Chọn caption tốt nhất trong danh sách candidates sau khi chuẩn hóa điểm theo độ dài.
    best_sequence, _score = max(
        completed,
        key=lambda item: _length_normalized_score(
            item[1], item[0].size(0) - 1, length_penalty
        ),
    )

    return vocab.decode(best_sequence.tolist())