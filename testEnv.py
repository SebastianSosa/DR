from environment import CustomEnv
env = CustomEnv()
episodes = 2

for episode in range(episodes):
	done = False
	obs = env.reset()
	while done == False:#not done:
		random_action = env.action_space.sample()
		print("action",random_action)
		obs, reward, done, info = env.step(random_action)
		print('reward',reward)
		print("done", done)
