# RNNimport numpy as np


data = [("I love deep", "learning"),
        ("deep learning is", "fun"),
        ("I enjoy deep", "learning"),
        ("learning is very", "powerful")]


vocab = list(set(" ".join(x[0] + " " + x[1] for x in data).split()))
word_to_idx = {w: i for i, w in enumerate(vocab)}
idx_to_word = {i: w for w, i in word_to_idx.items()}
vocab_size = len(vocab)


def one_hot(idx, size):
    vec = np.zeros(size)
    vec[idx] = 1
    return vec


hidden_size = 10
learning_rate = 0.01

Wxh = np.random.randn(hidden_size, vocab_size) * 0.01
Whh = np.random.randn(hidden_size, hidden_size) * 0.01
Why = np.random.randn(vocab_size, hidden_size) * 0.01
bh = np.zeros((hidden_size, 1))
by = np.zeros((vocab_size, 1))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


for epoch in range(1000):
    total_loss = 0
    for input_seq, target_word in data:
        words = input_seq.split()
        inputs = [one_hot(word_to_idx[w], vocab_size).reshape(-1, 1) for w in words]
        target = word_to_idx[target_word]

        # -------- Forward Pass --------
        hs = {}  # لحفظ الحالات المخفية
        hs[-1] = np.zeros((hidden_size, 1))  # الحالة الأولية
        for t in range(len(inputs)):
            hs[t] = np.tanh(np.dot(Wxh, inputs[t]) + np.dot(Whh, hs[t-1]) + bh)

      
        y_hat = softmax(np.dot(Why, hs[len(inputs)-1]) + by)

       
        loss = -np.log(y_hat[target])
        total_loss += loss

        # -------- Backward Pass --------
        dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
        dbh, dby = np.zeros_like(bh), np.zeros_like(by)
        dh_next = np.zeros_like(hs[0])

        dy = y_hat
        dy[target] -= 1  # مشتقة الـ softmax مع cross-entropy
        dWhy += np.dot(dy, hs[len(inputs)-1].T)
        dby += dy
        dh = np.dot(Why.T, dy) + dh_next

        for t in reversed(range(len(inputs))):
            dtanh = (1 - hs[t] ** 2) * dh
            dbh += dtanh
            dWxh += np.dot(dtanh, inputs[t].T)
            dWhh += np.dot(dtanh, hs[t-1].T)
            dh = np.dot(Whh.T, dtanh)

        for param, dparam in zip([Wxh, Whh, Why, bh, by],
                                 [dWxh, dWhh, dWhy, dbh, dby]):
            param -= learning_rate * dparam

    if epoch % 100 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss[0]:.4f}")


def predict_next(word_seq):
    inputs = [one_hot(word_to_idx[w], vocab_size).reshape(-1, 1) for w in word_seq.split()]
    h = np.zeros((hidden_size, 1))
    for x in inputs:
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = softmax(np.dot(Why, h) + by)
    pred_idx = np.argmax(y)
    return idx_to_word[pred_idx]

print("Test: I love deep →", predict_next("I love deep"))
print("Test: deep learning is →", predict_next("deep learning is"))
