import numpy as np

TRAINING_RATE = 0.001
GAMMA = 0.8
BATCH_SIZE = 100


def update_q_function(model, transitions, last_action):
    memory = np.array(transitions, dtype=object)
    action_batch_indices = np.where(memory[:, 1] == last_action)
    action_batch = memory[action_batch_indices]

    if len(action_batch) > BATCH_SIZE:
        action_batch = action_batch[np.random.randint(action_batch.shape[0], size=BATCH_SIZE), :]

    return update_gradient(model, action_batch, last_action)


def update_gradient(model, batch, last_action):
    gradient_updates = []

    for index, transition in enumerate(batch):
        old_features, action, new_features, reward = transition

        old_features = np.array(old_features)
        new_features = np.array(new_features)

        temporal_difference = calculate_temporal_difference(model, new_features, reward)
        gradient_update = np.dot(old_features.T, temporal_difference - np.dot(old_features, model[action]))
        gradient_updates.append(gradient_update)

    gradient_sum = np.sum(gradient_updates, axis=0)
    model[last_action] = model[last_action] + np.dot(TRAINING_RATE / len(batch), gradient_sum)
    return model


def calculate_temporal_difference(model, features, reward):
    return reward + GAMMA * calculate_q_max(model, features)


def calculate_q_max(model, features):
    possible_q_values = [np.dot(features, beta_action) for beta_action in model.values()]
    return np.max(possible_q_values)


def get_best_action(model, features):
    possible_q_values = [np.dot(features, beta) for beta in model.values()]
    best_index = int(np.argmax(possible_q_values))
    actions = list(model.keys())
    return actions[best_index]
