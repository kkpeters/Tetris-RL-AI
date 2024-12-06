package src.pas.tetris.agents;


// SYSTEM IMPORTS
import java.util.Iterator;
import java.util.List;
import java.util.Random;


// JAVA PROJECT IMPORTS
import edu.bu.tetris.agents.QAgent;
import edu.bu.tetris.agents.TrainerAgent.GameCounter;
import edu.bu.tetris.game.Board;
import edu.bu.tetris.game.Game.GameView;
import edu.bu.tetris.game.minos.Mino;
import edu.bu.tetris.linalg.Matrix;
import edu.bu.tetris.nn.Model;
import edu.bu.tetris.nn.LossFunction;
import edu.bu.tetris.nn.Optimizer;
import edu.bu.tetris.nn.models.Sequential;
import edu.bu.tetris.nn.layers.Dense; // fully connected layer
import edu.bu.tetris.nn.layers.ReLU;  // some activations (below too)
import edu.bu.tetris.nn.layers.Tanh;
import edu.bu.tetris.nn.layers.Sigmoid;
import edu.bu.tetris.training.data.Dataset;
import edu.bu.tetris.utils.Pair;


public class TetrisQAgent
    extends QAgent
{

    public static final double EXPLORATION_PROB = 0.05;

    private Random random;

    public TetrisQAgent(String name)
    {
        super(name);
        this.random = new Random(12345); // optional to have a seed
    }

    public Random getRandom() { return this.random; }

    @Override
    public Model initQFunction()
    {
        // System.out.println("initQFunction called!");
        // build a single-hidden-layer feedforward network
        // this example will create a 3-layer neural network (1 hidden layer)
        // in this example, the input to the neural network is the
        // image of the board unrolled into a giant vector
        //final int numPixelsInImage = Board.NUM_ROWS * Board.NUM_COLS;
        /*
        final int hiddenDim = 2 * numPixelsInImage;
        final int outDim = 1;
        */

        final int inputSize = 5;
        final int hiddenDim = 10;
        final int outDim = 1;


        Sequential qFunction = new Sequential();
        // qFunction.add(new Dense(numPixelsInImage, hiddenDim));
        qFunction.add(new Dense(inputSize, hiddenDim));
        // qFunction.add(new Tanh());
        qFunction.add(new ReLU());
        qFunction.add(new Dense(hiddenDim, outDim));

        return qFunction;
    }

    /**
     * TODO:
        This function is for you to figure out what your features
        are. This should end up being a single row-vector, and the
        dimensions should be what your qfunction is expecting.
        One thing we can do is get the grayscale image
        where squares in the image are 0.0 if unoccupied, 0.5 if
        there is a "background" square (i.e. that square is occupied
        but it is not the current piece being placed), and 1.0 for
        any squares that the current piece is being considered for.
        
        We can then flatten this image to get a row-vector, but we
        can do more than this! Try to be creative: how can you measure the
        "state" of the game without relying on the pixels? If you were given
        a tetris game midway through play, what properties would you look for?

        Possible factors:
            Number of Holes
            Bumpiness
            Height
            Looking ahead at the next three pieces
     */
    @Override
    public Matrix getQFunctionInput(final GameView game,
                                    final Mino potentialAction)
    {
        // matrix to store our features
        Matrix features = Matrix.zeros(1, 5); 
        double heightPenalty = 0.0;
        double bumpiness = 0.0; 
        double holePenalty = 0.0;
        double clearedLinesBonus = 0.0;

        Matrix flattenedImage = null;
        try
        {
            // flattenedImage = game.getGrayscaleImage(potentialAction).flatten();
            flattenedImage = game.getGrayscaleImage(potentialAction);
        } catch(Exception e)
        {
            e.printStackTrace();
            System.exit(-1);
        }

        int missMino = -1;
        Mino.MinoType thisMino = potentialAction.getType(); 

        if (thisMino == Mino.MinoType.valueOf("I")) {
            missMino = 0;
        }
        if (thisMino == Mino.MinoType.valueOf("J")) {
            missMino = 1;
        }
        if (thisMino == Mino.MinoType.valueOf("L")) {
            missMino = 2;
        }
        if (thisMino == Mino.MinoType.valueOf("O")) {
            missMino = 3;
        }
        if (thisMino == Mino.MinoType.valueOf("S")) {
            missMino = 4;
        }
        if (thisMino == Mino.MinoType.valueOf("T")) {
            missMino = 5;
        }
        if (thisMino == Mino.MinoType.valueOf("Z")) {
            missMino = 6;
        }




        /*
        System.out.println(flattenedImage.getShape().getNumRows());
        System.out.println(flattenedImage.getShape().getNumCols());
        */
        int[] columnHeights = new int[flattenedImage.getShape().getNumCols()];
        for (int i = 0; i < columnHeights.length; i++) {
            columnHeights[i] = -1;
        }


        for (int row = 0; row < flattenedImage.getShape().getNumRows(); row++) {
            boolean lineCleared = true; 
            for (int col = 0; col < flattenedImage.getShape().getNumCols(); col++) {
                // double cell = board.isCoordinateOccupied(row, col) ? 1.0 : 0.0;

                // cell is occupied
                if(flattenedImage.get(row, col) == 1.0 || flattenedImage.get(row, col) == 0.5) {
                    if (columnHeights[col] == -1) {
                        columnHeights[col] = row;
                    }
                    lineCleared = false;
                }

                // Calculate hole penalty (empty space below filled squares)
                else if (flattenedImage.get(row, col) == 0.0) {
                    if (columnHeights[col] != -1) {
                        holePenalty += 1.0; // Increment hole penalty for each hole
                        //print if there is a hole
                        // System.out.println("Hole at row=" + row + " col=" + col);
                        //
                    }
                }
            }
            if (lineCleared) {
                clearedLinesBonus += 1.0; // Award bonus for each line cleared
            }
        }
        /*
        for (int col = 0; col < Board.NUM_COLS; col++) {
            boolean lineCleared = true; 
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                System.out.println("row: " + row + ", col: " + col);
                double cell = 0.0; 
                if (board.isCoordinateOccupied(col, row)) {
                    isBoardEmpty = false;
                    // Set the height of the column when a block is found
                    columnHeights[col] = Board.NUM_ROWS - row; 
                    //System.out.println("Board occupied at cell: " + col + ", " + row);
                    lineCleared = false; 
                    // If any cell in the row is empty, the line is not cleared
                }
                else {
                    //cell = 0.0;
                    holePenalty += 1.0; // Increment hole penalty for each hole
                    //print if there is a hole
                    //System.out.println("Hole at row=" + row + " col=" + col);
                    if(row == Board.NUM_ROWS - 1 && lineCleared && !isBoardEmpty){
                        completedRows += 1;
                    }

                }
            }
        }
        */
        // calculate height penalty
        for (int i = 0; i < Board.NUM_COLS; i++) {
            if (columnHeights[i] != -1) {
                heightPenalty += columnHeights[i];
            }
        }
        // calculate bumpiness
        for (int col = 0; col < Board.NUM_COLS - 1; col++) {
            int height1 = 0;
            int height2 = 0;
          
            if (columnHeights[col] != -1) {
                height1 = columnHeights[col];
            }

            if (columnHeights[col + 1] != -1) {
                height2 = columnHeights[col + 1];
            }
            bumpiness += Math.abs(height1 - height2); // Add the difference in heights between adjacent columns
        }




        features.set(0, 0, heightPenalty);
        features.set(0, 1, holePenalty);
        features.set(0, 2, clearedLinesBonus);
        features.set(0, 3, bumpiness);
        features.set(0, 4, missMino);




        return features;
    }

    /**
     * TODO:
     * This method is used to decide if we should follow our current policy
     * (i.e. our q-function), or if we should ignore it and take a random action
     * (i.e. explore).
     *
     * Remember, as the q-function learns, it will start to predict the same "good" actions
     * over and over again. This can prevent us from discovering new, potentially even
     * better states, which we want to do! So, sometimes we should ignore our policy
     * and explore to gain novel experiences.
     *
     * The current implementation chooses to ignore the current policy around 5% of the time.
     * While this strategy is easy to implement, it often doesn't perform well and is
     * really sensitive to the EXPLORATION_PROB. I would recommend devising your own
     * strategy here.
     */
    @Override
    public boolean shouldExplore(final GameView game,
                                 final GameCounter gameCounter)
    {
        int currGame = (int)gameCounter.getCurrentGameIdx();

        double highestExploreProb = 1.0;
        double decayRate = 0.99 - (currGame * 0.01);
        double xyz = highestExploreProb - EXPLORATION_PROB;

        double currTurn = (double)gameCounter.getCurrentMoveIdx() * (xyz) / decayRate;

        // later on in the game we want to use knowledge we already know rather than taking a random move that may not be good
        double currentExplorationProb = Math.max(EXPLORATION_PROB, highestExploreProb - currTurn);

        // System.out.println("phaseIdx=" + gameCounter.getCurrentPhaseIdx() + "\tgameIdx=" + gameCounter.getCurrentGameIdx());
        return this.getRandom().nextDouble() <= currentExplorationProb;

    }

    /**
     * TODO:
     * This method is a counterpart to the "shouldExplore" method. Whenever we decide
     * that we should ignore our policy, we now have to actually choose an action.
     *
     * You should come up with a way of choosing an action so that the model gets
     * to experience something new. The current implemention just chooses a random
     * option, which in practice doesn't work as well as a more guided strategy.
     * I would recommend devising your own strategy here.
     */
    @Override
    public Mino getExplorationMove(final GameView game)
    {
        int numMinoPositions = game.getFinalMinoPositions().size();
        // Matrix qFunctions = Matrix.zeros(1, numMinoPositions);
        double[] qFunctions = new double[numMinoPositions];

        for(int i = 0; i < numMinoPositions; i++){
            Matrix cur = this.getQFunctionInput(game, game.getFinalMinoPositions().get(i));
            // gets the Q function
            try {
                qFunctions[i] = Math.exp(this.initQFunction().forward(cur).get(0, 0));
            }
            catch (Exception e) {
                e.printStackTrace();
                System.exit(-1);
            }
        }
        int minValPos = 0;
        double minVal = Double.POSITIVE_INFINITY;

        // sum of all the q values
        double qSum = 0.0;
        for (int i = 0; i < numMinoPositions; i++) {
            qSum += qFunctions[i];
        }

        // normalizing
        double[] outcome = new double[numMinoPositions];
        for (int i = 0; i < numMinoPositions; i++) {
            outcome[i] = qFunctions[i] / qSum;
        }
        for (int i = 0; i < numMinoPositions; i++) {
            if (outcome[i] < minVal) {
                minVal = outcome[i];
                minValPos = i;
            }
        }
        return game.getFinalMinoPositions().get(minValPos);

        /*
        Mino bestMino = getBestActionAndQValue(game).getFirst();
        return bestMino;
        */









        /*
        int randIdx = this.getRandom().nextInt(game.getFinalMinoPositions().size());
        return game.getFinalMinoPositions().get(randIdx);
        */
    }

    /**
     * This method is called by the TrainerAgent after we have played enough training games.
     * In between the training section and the evaluation section of a phase, we need to use
     * the exprience we've collected (from the training games) to improve the q-function.
     *
     * You don't really need to change this method unless you want to. All that happens
     * is that we will use the experiences currently stored in the replay buffer to update
     * our model. Updates (i.e. gradient descent updates) will be applied per minibatch
     * (i.e. a subset of the entire dataset) rather than in a vanilla gradient descent manner
     * (i.e. all at once)...this often works better and is an active area of research.
     *
     * Each pass through the data is called an epoch, and we will perform "numUpdates" amount
     * of epochs in between the training and eval sections of each phase.
     */
    @Override
    public void trainQFunction(Dataset dataset,
                               LossFunction lossFunction,
                               Optimizer optimizer,
                               long numUpdates)
    {
        for(int epochIdx = 0; epochIdx < numUpdates; ++epochIdx)
        {
            dataset.shuffle();
            Iterator<Pair<Matrix, Matrix> > batchIterator = dataset.iterator();

            while(batchIterator.hasNext())
            {
                Pair<Matrix, Matrix> batch = batchIterator.next();

                try
                {
                    Matrix YHat = this.getQFunction().forward(batch.getFirst());

                    optimizer.reset();
                    this.getQFunction().backwards(batch.getFirst(),
                                                  lossFunction.backwards(YHat, batch.getSecond()));
                    optimizer.step();
                } catch(Exception e)
                {
                    e.printStackTrace();
                    System.exit(-1);
                }
            }
        }
    }

    /**
     * TODO:
     * This method is where you will devise your own reward signal. Remember, the larger
     * the number, the more "pleasurable" it is to the model, and the smaller the number,
     * the more "painful" to the model.
     *
     * This is where you get to tell the model how "good" or "bad" the game is.
     * Since you earn points in this game, the reward should probably be influenced by the
     * points, however this is not all. In fact, just using the points earned this turn
     * is a **terrible** reward function, because earning points is hard!!
     *
     * I would recommend you to consider other ways of measuring "good"ness and "bad"ness
     * of the game. For instance, the higher the stack of minos gets....generally the worse
     * (unless you have a long hole waiting for an I-block). When you design a reward
     * signal that is less sparse, you should see your model optimize this reward over time.
     */
    @Override
    public double getReward(final GameView game)
    {
        Board board = game.getBoard();
        /*
        System.out.print("Num_cols: ");
        System.out.println(Board.NUM_COLS);
        System.out.print("Num_rows: ");
        System.out.println(Board.NUM_ROWS);
        */

        /*
        if (game.didAgentLose()) {
            return -500.0; // Large negative reward for losing
        }
        */
        double reward = 0.0;


        double heightPenalty = 0.0;
        double bumpiness = 0.0; 
        double holePenalty = 0.0;

        boolean isBoardEmpty = true;
        int completedRows = 0;

        int[] columnHeights = new int[Board.NUM_COLS];
        for (int i = 0; i < Board.NUM_COLS; i++) {
            columnHeights[i] = -1;
            //System.out.println(columnHeights[i]);
        }

        boolean prevClearedRow = false;
        // double twoRowWeight = 1.0;

        for (int col = 0; col < Board.NUM_COLS; col++) {
            boolean lineCleared = true; 
            for (int row = 0; row < Board.NUM_ROWS; row++) {
                //System.out.println("row: " + row + ", col: " + col);
                double cell = 0.0; 
                if (board.isCoordinateOccupied(col, row)) {
                    isBoardEmpty = false;
                    // Set the height of the column when a block is found
                    columnHeights[col] = Board.NUM_ROWS - row; 
                    //System.out.println("Board occupied at cell: " + col + ", " + row);
                    lineCleared = false; 
                    // If any cell in the row is empty, the line is not cleared
                }
                else {
                    //cell = 0.0;
                    holePenalty += 1.0; // Increment hole penalty for each hole
                    //print if there is a hole
                    //System.out.println("Hole at row=" + row + " col=" + col);
                    if(row == Board.NUM_ROWS - 1 && lineCleared && !isBoardEmpty){
                        completedRows += 1;
                        //prevClearedRow = true;
                        // if previous turn had cleared a row, then weight this more
                        //if (!prevClearedRow){
                        //    twoRowWeight = 50;
                        //    prevClearedRow = false;
                        //}
                    }

                }
            }
        }
        // calculate height penalty
        for (int i = 0; i < Board.NUM_COLS; i++) {
            if (columnHeights[i] != -1) {
                heightPenalty += columnHeights[i];
                //heightPenalty += columnHeights[i];
            }
            //else {
            //    heightPenalty = 0;
            //}
            //heightPenalty += columnHeights[i] != -1 ? columnHeights[i] : 0;
        }
        // calculate bumpiness
            /*
        double avgHeight = 0.0;
        for (int col = 0; col < Board.NUM_COLS - 1; col++) {
            avgHeight += columnHeights[col];
            //int height1 = columnHeights[col] != -1 ? columnHeights[col] : 0;
            int height1 = 0;
            int height2 = 0;
            if (columnHeights[col] != -1) {
                height1 = columnHeights[col];
            }
            // int height2 = columnHeights[col + 1] != -1 ? columnHeights[col + 1] : 0;
            if (columnHeights[col + 1] != -1) {
                height2 = columnHeights[col + 1];
            }
            bumpiness += Math.abs(height1 - height2); // Add the difference in heights between adjacent columns
        }
        avgHeight /= Board.NUM_COLS
            */
        double totalHeight = 0;
        double currMaxHeight = -1;
        for (int col = 0; col < Board.NUM_COLS; col++) {
            //System.out.println("Column height: " + columnHeights[col]);
            totalHeight += columnHeights[col];
            /*
            if (columnHeights[col] > currMaxHeight) {
                currMaxHeight = columnHeights[col];
                //System.out.println("Curr Max height: " + currMaxHeight);
            }
            */











            // TODO: issue with total height, if one column is super tall, the total height is still low, but that column will lose the game bc it's too high
            // also i dont think this is working in general











            //int height1 = columnHeights[col] != -1 ? columnHeights[col] : 0;
            /*int height1 = 0;
            int height2 = 0;
          
            if (columnHeights[col] != -1) {
                height1 = columnHeights[col];
            }

            // int height2 = columnHeighWts[col + 1] != -1 ? columnHeights[col + 1] : 0;
            if (columnHeights[col + 1] != -1) {
                height2 = columnHeights[col + 1];
            }
            bumpiness += Math.abs(height1 - height2); // Add the difference in heights between adjacent columns*/

        }
        //System.out.println("---------");

        // Encourage column height uniformity (reduce variance)
        double meanHeight = totalHeight / (double) Board.NUM_COLS;
        double heightVariance = 0;
        for (int i = 0; i < columnHeights.length; i ++) {
            heightVariance += Math.pow(columnHeights[i] - meanHeight, 2);
        }
        heightVariance /= Board.NUM_COLS;


        int turnScore = game.getScoreThisTurn();

        // gives a heigher reward if it clears more than 1 line bc that actually scores more points
        if (completedRows >= 2) {
            reward += completedRows * 70.0;
        }
        else {
            reward += completedRows * 40.0;
        }
        /*
        if (currMaxHeight >= 10) {
            reward -= currMaxHeight * 10.0;
        }
        */
        
        /*
        try {
            // Sleep for 2 seconds (2000 milliseconds)
            Thread.sleep(2000);
        } catch (InterruptedException e) {
            // Handle the exception if the thread is interrupted
            System.out.println("Sleep interrupted!");
        }
        */
        

        reward -= heightPenalty * 20.0;
        reward -= heightVariance * 50.0;
        reward -= holePenalty * 30.0;
        reward += turnScore * 40.0;
        return reward;
    }
}
