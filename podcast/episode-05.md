# Episode 05: Every card on Steam, and a stronger model overnight

Last time, the new model was a portfolio. It is a set of specialists, one built around Witch, one around Militia, one around Gardens, and so on. When a game starts, it tries every specialist on the cards that were dealt, and the best one plays. But nine cards were still blocked. The autopilot could not answer the questions Steam asks when you play them, so no model was allowed to buy them. You asked me to fix that, and then to keep improving the model through the night.

The fix had two halves. The first half was the model side. Every card that asks you something now has its own question in the protocol. Remodel asks what to trash and what to gain. Cellar asks what to discard. Sentry asks what to do with the top two cards of your deck. Each answer comes from the same code the model trained with. A test plays three hundred games in the engine, and at every one of these choices it checks that the answer the autopilot would get is exactly what the model did in training. They matched every time.

The second half was the screen. I had to learn how Steam actually asks. So the autopilot played four games against the easiest computer player, with a test model that buys all the new cards. Every time it met a screen it did not know, it saved a picture and stopped. Then I taught it that screen and let it continue. Gaining a card is a click on a glowing pile. Cellar is a choice of cards and then OK, or a button called Decline. Harbinger and Vassal show a small window to pick from. Library has a button called Set Aside. Sentry was the surprise. You have to drag the cards into zones called trash, discard and deck. So the autopilot learned to drag.

Those games found several old bugs too. A Militia attack shows a counter, zero of two, and the autopilot read that zero as the number of cards to discard. When an empty pile was not noticed, Workshop clicked it fifty eight times in a row. And the model itself, when attacked by Militia, kept a Province in hand and threw away a Gold. All of these are fixed.

Then came a real game, with all cards allowed, against the easy computer player. The portfolio looked at the kingdom and picked its Gardens and Workshop specialist, which could never be used before. It won, thirty four to thirty three.

Overnight came the second round of training. In the first round, every specialist learned against three simple scripted opponents. This time each one also had to beat three strong specialists from the first round. Twenty six specialists trained this way, about fifteen minutes each. Together they form portfolio version two, now the default.

The best test is a match. Both portfolios look at the same random kingdom, each picks its own specialist, and they play each other. Over two hundred kingdoms, version two wins fifty seven percent of the games. Against the simple opponents both are close to the limit, eighty one percent against eighty.

One experiment failed. I gave the generalist, the specialist for any kingdom, four times as much training. It did not improve at all after the first quarter. So its limit is the shape of its strategy, not training time.

Two rare screens were never seen yet. One is revealing a Moat when attacked, and the other is choosing which treasure a Bandit takes. Next is a third round, where the new specialists train against each other, and then real games against the hard computer player.
