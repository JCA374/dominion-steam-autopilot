# Episode 04: One engine for every model, and how the pieces fit

Since the last episode, the autopilot played a full campaign on Steam. Eleven games were finished. It won five and lost six. Against the two easiest computer players it won four of five, and the one loss was caused by a bug in the autopilot, which is now fixed. Against the medium player it won one of three. Against the hard player it lost all three, by two, eight and eighteen points.

The reason was clear from the games. The hard player ends the game fast, in about twenty turns. Our model builds a slow money deck, and it ignores most kingdom cards. In a typical game, about six of the ten kingdom cards were cards it had never learned.

That led to a bigger finding. The simulator the strategies were evolved in did not play quite the same game as Steam. It used twelve Provinces in a two player game, and Steam uses eight. It knew fifteen of the twenty six kingdom cards. Every simulated game stopped at turn forty. And a tie counted as half a win, while Steam gives the win to the player who took fewer turns. So the models were trained for a slightly different game.

Today that was fixed at the root. The game itself is now its own project, called the engine. It plays Dominion the way Steam does, with all twenty six cards. It exists twice, once in Python, which is easy to read, and once in C, which is about sixty times faster for training. Both versions shuffle the same way and ask the same questions in the same order. A test plays fifteen hundred random games in both and compares every single decision. There were no differences.

Here is how the whole setup fits together. Think of three pieces in a row.

The first piece is the engine. It is the rules and nothing else. Every choice in a game, like which action to play, what to buy, or what to discard to a Militia, is asked as one question with a list of answers. Anything that can answer those questions can play. That could be a simple bot, an evolved strategy, you in the game window, or later a neural network.

The second piece is a trainer. Right now there is one, the genetic algorithm. It plays millions of games on the engine and evolves a strategy. When a strategy is good, it is released as a model. A model is just a folder. It holds the evolved strategy, the kingdom it was trained on, its key cards, and a benchmark score. The key cards are the kingdom cards it is built around. The benchmark is how often it beats three fixed opponents on two hundred random kingdoms. All the models were benchmarked again today on the new engine. The strongest one is still the model built around Witch and Market, at about eighty percent.

The third piece is the autopilot. It reads the Steam screen and clicks. When a game starts, it reads the ten kingdom cards from the game's save file. Then a selector picks the model for that kingdom. First choice is a model trained on exactly those cards. Next is a model whose key cards were all dealt, with the best benchmark. A model only ever buys cards it was trained with, so the right model matters.

On disk, this is three folders. The engine folder holds the game. The trainers folder holds one folder per kind of model, and the genetic algorithm lives in one called ga v one. A future neural network would get its own folder right next to it. The autopilot folder holds the screen reading, the control panel, and this website. Each trainer pins the exact version of the engine it trained on.

One honest limit. For nine of the new cards, like Cellar and Remodel, the autopilot can't yet answer the questions Steam asks when you play them. So for now, no model is allowed to buy those nine.

What's next is clear. Train new models on the new engine, with the real rules and all the cards. Then send them back into the campaign, and see if they can finally beat the hard player.
