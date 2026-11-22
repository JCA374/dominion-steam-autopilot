# Episode 10: Why the win chance jumps around

You noticed that the advisor's win chance swings too much. In one turn of today's game it went from twenty nine percent to eighty six and back. So this episode is about where that number comes from, why it moves, and what I changed.

First, how it is calculated. The advisor does not count points or simulate the rest of the game. It asks the neural network from the reinforcement learning project. That network has two outputs. One is a probability for each move. The other is a single number for how good the position is, between minus one and one. The advisor turns that number into a percentage. Minus one is zero percent, one is a hundred.

The network learned that number from many training games, against itself and against other bots. After each game, the positions it had seen were nudged toward the result. So the win chance is the network's gut feeling, learned from its own games. Nobody has checked yet that seventy percent really means winning seven games in ten.

Now, why it jumps. I took the real position from today's game and changed one thing at a time.

The first surprise is the question itself. The network is always asked about a decision, and the kind of decision is one of its inputs. On the very same position, a question about which card to play gave ninety percent. A question about what to buy gave eighty six. A question about what to discard to a Militia gave seventy six. That is sixteen points from nothing but the kind of question on screen.

The second is the score. The advisor gives the network the score it reads from the screen. Today that score and the counted cards disagreed. That alone moved the number by eight points.

The third is real. Coins in hand matter. With nothing in hand the network said eighty six. With eight coins, ninety six. And which five cards you drew matters too. Thirty different hands from the same deck ranged from eighty two to ninety seven.

The fourth is plain mistakes. This morning a Province pile was misread as empty. The network was told the game was nearly over with you far ahead, and it said one hundred percent. That one is fixed now.

So what did I change? My first idea was to ask the same question every time and smooth the number within a turn. I tested that on a recorded game. It did not help. The median swing within a turn went from fourteen points to seventeen. Most of the movement is the network reacting to real things, like a card played or coins appearing.

So now the big win chance at the top is taken once, at the start of your turn. It uses your dealt hand, no coins yet, and always the same question. It stays put until your next turn. The live number has not gone away. It now sits with each suggestion, as the win chance after this move. So you can still see that a good draw or a big buy helped. On the same recorded game, the biggest swing of the headline within a turn fell from twenty six points to fourteen. What is left is mostly one step, when your new hand arrives.

What I don't know yet is whether the number is honest. A network can be confident and wrong. The next step is to check it. I can let the engine play a few thousand games and record the network's win chance at each turn. Then I group the turns by what it predicted, and see how often those games were really won. If seventy percent turns out to mean fifty, I can correct it with a simple rescaling, so the number means what it says.
