# Episode 07: Is the GA undertrained?

You asked a fair question. The genetic algorithm behind the portfolio is the baseline every newer model is measured against. The reinforcement learning network beats it about seventy percent of the time. So maybe the baseline is weak only because it never trained long enough. Today I tested that, and the answer is mostly no.

Here is what I looked at first. Each specialist in portfolio version two trained for five hundred generations. Most of them had flat curves at the end. But a few were still climbing. Mine, Vassal, Workshop and Witch had all gained between three and five points in their last hundred generations. That looked like a model that stopped too early.

So I took the released specialists and kept training them, one thousand more generations each, with the same settings. Every twenty five generations the run saved its best genome. Then I measured each saved genome on games no run had ever seen. Four thousand games against the bots, and eight thousand head to head against the version two specialist it started from.

Witch did not move. After a thousand generations it won forty nine and a half percent against its old self. Mine won fifty and a half. Gardens stayed flat on the training scores the whole time. These specialists had already trained in an earlier round, so in total they have a thousand generations behind them, and they have stopped.

Two specialists did get better. Workshop won fifty two point seven percent against its old version, and Vassal fifty four point nine. Both also scored two to three points more against the bots. That is well outside the noise. What they have in common is that they were trained from scratch in round two, with only five hundred generations. So you were right about those two. They were undertrained.

But does that make the baseline stronger? I swapped the better Workshop and Vassal into the portfolio and played it against the original on three hundred fresh kingdoms. Fifty point one percent. No difference. The portfolio rarely picks those two, and the baseline the other models are measured against never uses them at all. That baseline mostly plays Witch, Militia, Gardens, Market and Bandit, and those are the ones that no longer improve.

Then I tried your other idea, training against the best model instead of only the bots. I plugged the reinforcement learning network into the genetic algorithm as an opponent, weighted twice as much as a bot. The Witch specialist trained against it for four hundred generations. It never found anything better. Against the network it scored forty one and a half percent, and the starting genome forty two. The net is a stronger teacher, but the student can't learn what it teaches.

So the lesson repeats what episode six suspected, now with a direct test. A buy list in three phases has a ceiling, and the main specialists have reached it. More generations and a stronger opponent both leave them where they are. The only real gains were in specialists that started late, and they barely matter to the portfolio.

I stopped all the training, as you asked, when it stopped paying off. The useful leftovers are a measurement script for specialists, a fix that lets the trainer use evolved opponents on the new Python version, and the two better specialists, which are not released. If the baseline is going to get stronger, it has to be a different kind of player, like the search from last episode, not more of the same evolution.
