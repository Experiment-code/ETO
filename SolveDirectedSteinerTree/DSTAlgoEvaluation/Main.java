import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.GFLAC2Algorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.GFLACAlgorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.RoosAlgorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.ShP2Algorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.ShPAlgorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.SteinerArborescenceApproximationAlgorithm;
import graphTheory.algorithms.steinerProblems.steinerArborescenceApproximation.WongAlgorithm;
import graphTheory.generators.RandomSteinerDirectedGraphGenerator2;
import graphTheory.generators.steinLib.STPDirectedGenerator;
import graphTheory.generators.steinLib.STPGenerator;
import graphTheory.generators.steinLib.STPUndirectedGenerator;
import graphTheory.graph.Arc;
import graphTheory.graph.DirectedGraph;
import graphTheory.graph.UndirectedGraph;
import graphTheory.graphDrawer.EnergyAnalogyGraphDrawer;
import graphTheory.instances.steiner.classic.SteinerDirectedInstance;
import graphTheory.instances.steiner.classic.SteinerUndirectedInstance;
import graphTheory.steinLib.STPTranslationException;
import graphTheory.steinLib.STPTranslator;
import graphTheory.steinLib.SteinLibInstancesGroups;
import graphTheory.utils.FileManager;
import graphTheory.utils.probabilities.BBernouilliLaw;
import graphTheory.utils.probabilities.BConstantLaw;

import java.awt.Color;
import java.io.File;
import java.util.HashSet;

/**
 * 
 * @author Watel Dimitri
 * 
 */
public class Main {

	
	public static void main(String[] args) {
	
			// Run this method to compute a small SteinerDirectedInstance, solve it and
			// display it on the screen.
//			exampleCreateInstanceAndDisplay();
			select_reuse_pairs();
//			select_reuse_pairs_old();
			
			// Run this method to create a random instance and display it on the screen
//			exampleCreateRandomInstanceAndDisplay();
			
			// Run one of those methods to compute a small SteinerUndirectedInstance
			// and transform it into a directed one : bidirected, acyclic or strongly connected
//			exampleTransformUndirectedIntoBidirected();
//			exampleTransformUndirectedIntoAcyclic();
//			exampleTransformUndirectedIntoStronglyConnected();
			
			// Run one of those methods to transform all the instances describe with the
			// STP format in the directory SteinLib/B into directed instances (bidirected
			// acyclic or strongly connected) and store them into their own directory.
//			exampleCreateBidirectedInstances();
//			exampleCreateAcyclicInstances();
//			exampleCreateSronglyConnectedInstances();
			
			// Run one of those methods to compute the GFLACAlgorithm over the previous
			// generated instances and show the results on standart output.
//			exampleLaunchBidirTest();
//			exampleLaunchAcyclicTest();
//			exampleLaunchStronglyTest();
			
		}

	/*------------------------------------------------------------------------
	 * 
	 * Example on how to use the Graph and SteinerDirectedInstance classes.
	 * 
	 *------------------------------------------------------------------------
	 */
	
	
	
	/**
	 * Create a Steiner instance and run an approximation over it. Finally draw
	 * the instance and the returned solution on the screen.
	 */
	
//	REPLACE THE 3 LINES BELOW WITH THE CONTENT FROM SelectiveReusePairs.java 
//	depthwise
	public static DirectedGraph get_dg1() {DirectedGraph dg = new DirectedGraph();for (int i = 1; i <= 22; i++)	dg.addVertice(i);dg.addDirectedEdges(1, 2);dg.addDirectedEdges(3, 2);dg.addDirectedEdges(12, 2);dg.addDirectedEdges(13, 2);dg.addDirectedEdges(14, 2);dg.addDirectedEdges(15, 2);dg.addDirectedEdges(16, 2);dg.addDirectedEdges(1, 3);dg.addDirectedEdges(2, 3);dg.addDirectedEdges(4, 3);dg.addDirectedEdges(12, 3);dg.addDirectedEdges(13, 3);dg.addDirectedEdges(14, 3);dg.addDirectedEdges(15, 3);dg.addDirectedEdges(16, 3);dg.addDirectedEdges(1, 4);dg.addDirectedEdges(3, 4);dg.addDirectedEdges(5, 4);dg.addDirectedEdges(6, 4);dg.addDirectedEdges(12, 4);dg.addDirectedEdges(13, 4);dg.addDirectedEdges(14, 4);dg.addDirectedEdges(15, 4);dg.addDirectedEdges(16, 4);dg.addDirectedEdges(17, 4);dg.addDirectedEdges(18, 4);dg.addDirectedEdges(19, 4);dg.addDirectedEdges(1, 5);dg.addDirectedEdges(4, 5);dg.addDirectedEdges(6, 5);dg.addDirectedEdges(12, 5);dg.addDirectedEdges(13, 5);dg.addDirectedEdges(14, 5);dg.addDirectedEdges(15, 5);dg.addDirectedEdges(16, 5);dg.addDirectedEdges(17, 5);dg.addDirectedEdges(18, 5);dg.addDirectedEdges(19, 5);dg.addDirectedEdges(1, 6);dg.addDirectedEdges(4, 6);dg.addDirectedEdges(5, 6);dg.addDirectedEdges(7, 6);dg.addDirectedEdges(13, 6);dg.addDirectedEdges(14, 6);dg.addDirectedEdges(15, 6);dg.addDirectedEdges(16, 6);dg.addDirectedEdges(17, 6);dg.addDirectedEdges(18, 6);dg.addDirectedEdges(19, 6);dg.addDirectedEdges(1, 7);dg.addDirectedEdges(6, 7);dg.addDirectedEdges(8, 7);dg.addDirectedEdges(13, 7);dg.addDirectedEdges(14, 7);dg.addDirectedEdges(15, 7);dg.addDirectedEdges(16, 7);dg.addDirectedEdges(17, 7);dg.addDirectedEdges(18, 7);dg.addDirectedEdges(19, 7);dg.addDirectedEdges(20, 7);dg.addDirectedEdges(21, 7);dg.addDirectedEdges(1, 8);dg.addDirectedEdges(7, 8);dg.addDirectedEdges(9, 8);dg.addDirectedEdges(14, 8);dg.addDirectedEdges(15, 8);dg.addDirectedEdges(16, 8);dg.addDirectedEdges(17, 8);dg.addDirectedEdges(18, 8);dg.addDirectedEdges(19, 8);dg.addDirectedEdges(20, 8);dg.addDirectedEdges(21, 8);dg.addDirectedEdges(1, 9);dg.addDirectedEdges(8, 9);dg.addDirectedEdges(10, 9);dg.addDirectedEdges(14, 9);dg.addDirectedEdges(15, 9);dg.addDirectedEdges(16, 9);dg.addDirectedEdges(17, 9);dg.addDirectedEdges(18, 9);dg.addDirectedEdges(19, 9);dg.addDirectedEdges(20, 9);dg.addDirectedEdges(21, 9);dg.addDirectedEdges(22, 9);dg.addDirectedEdges(1, 10);dg.addDirectedEdges(9, 10);dg.addDirectedEdges(15, 10);dg.addDirectedEdges(16, 10);dg.addDirectedEdges(18, 10);dg.addDirectedEdges(19, 10);dg.addDirectedEdges(20, 10);dg.addDirectedEdges(21, 10);dg.addDirectedEdges(22, 10);dg.addDirectedEdges(1, 11);dg.addDirectedEdges(16, 11);dg.addDirectedEdges(19, 11);dg.addDirectedEdges(21, 11);dg.addDirectedEdges(22, 11);dg.addDirectedEdges(1, 12);dg.addDirectedEdges(2, 12);dg.addDirectedEdges(3, 12);dg.addDirectedEdges(4, 12);dg.addDirectedEdges(5, 12);dg.addDirectedEdges(13, 12);dg.addDirectedEdges(14, 12);dg.addDirectedEdges(15, 12);dg.addDirectedEdges(16, 12);dg.addDirectedEdges(1, 13);dg.addDirectedEdges(2, 13);dg.addDirectedEdges(3, 13);dg.addDirectedEdges(4, 13);dg.addDirectedEdges(5, 13);dg.addDirectedEdges(6, 13);dg.addDirectedEdges(7, 13);dg.addDirectedEdges(12, 13);dg.addDirectedEdges(14, 13);dg.addDirectedEdges(15, 13);dg.addDirectedEdges(16, 13);dg.addDirectedEdges(1, 14);dg.addDirectedEdges(2, 14);dg.addDirectedEdges(3, 14);dg.addDirectedEdges(4, 14);dg.addDirectedEdges(5, 14);dg.addDirectedEdges(6, 14);dg.addDirectedEdges(7, 14);dg.addDirectedEdges(8, 14);dg.addDirectedEdges(9, 14);dg.addDirectedEdges(12, 14);dg.addDirectedEdges(13, 14);dg.addDirectedEdges(15, 14);dg.addDirectedEdges(16, 14);dg.addDirectedEdges(17, 14);dg.addDirectedEdges(1, 15);dg.addDirectedEdges(2, 15);dg.addDirectedEdges(3, 15);dg.addDirectedEdges(4, 15);dg.addDirectedEdges(5, 15);dg.addDirectedEdges(6, 15);dg.addDirectedEdges(7, 15);dg.addDirectedEdges(8, 15);dg.addDirectedEdges(9, 15);dg.addDirectedEdges(10, 15);dg.addDirectedEdges(12, 15);dg.addDirectedEdges(13, 15);dg.addDirectedEdges(14, 15);dg.addDirectedEdges(16, 15);dg.addDirectedEdges(17, 15);dg.addDirectedEdges(18, 15);dg.addDirectedEdges(20, 15);dg.addDirectedEdges(1, 16);dg.addDirectedEdges(2, 16);dg.addDirectedEdges(3, 16);dg.addDirectedEdges(4, 16);dg.addDirectedEdges(5, 16);dg.addDirectedEdges(6, 16);dg.addDirectedEdges(7, 16);dg.addDirectedEdges(8, 16);dg.addDirectedEdges(9, 16);dg.addDirectedEdges(10, 16);dg.addDirectedEdges(11, 16);dg.addDirectedEdges(12, 16);dg.addDirectedEdges(13, 16);dg.addDirectedEdges(14, 16);dg.addDirectedEdges(15, 16);dg.addDirectedEdges(17, 16);dg.addDirectedEdges(18, 16);dg.addDirectedEdges(19, 16);dg.addDirectedEdges(20, 16);dg.addDirectedEdges(21, 16);dg.addDirectedEdges(22, 16);dg.addDirectedEdges(1, 17);dg.addDirectedEdges(4, 17);dg.addDirectedEdges(5, 17);dg.addDirectedEdges(6, 17);dg.addDirectedEdges(7, 17);dg.addDirectedEdges(8, 17);dg.addDirectedEdges(9, 17);dg.addDirectedEdges(14, 17);dg.addDirectedEdges(15, 17);dg.addDirectedEdges(16, 17);dg.addDirectedEdges(18, 17);dg.addDirectedEdges(19, 17);dg.addDirectedEdges(1, 18);dg.addDirectedEdges(4, 18);dg.addDirectedEdges(5, 18);dg.addDirectedEdges(6, 18);dg.addDirectedEdges(7, 18);dg.addDirectedEdges(8, 18);dg.addDirectedEdges(9, 18);dg.addDirectedEdges(10, 18);dg.addDirectedEdges(15, 18);dg.addDirectedEdges(16, 18);dg.addDirectedEdges(17, 18);dg.addDirectedEdges(19, 18);dg.addDirectedEdges(20, 18);dg.addDirectedEdges(1, 19);dg.addDirectedEdges(4, 19);dg.addDirectedEdges(5, 19);dg.addDirectedEdges(6, 19);dg.addDirectedEdges(7, 19);dg.addDirectedEdges(8, 19);dg.addDirectedEdges(9, 19);dg.addDirectedEdges(10, 19);dg.addDirectedEdges(11, 19);dg.addDirectedEdges(16, 19);dg.addDirectedEdges(17, 19);dg.addDirectedEdges(18, 19);dg.addDirectedEdges(20, 19);dg.addDirectedEdges(21, 19);dg.addDirectedEdges(22, 19);dg.addDirectedEdges(1, 20);dg.addDirectedEdges(7, 20);dg.addDirectedEdges(8, 20);dg.addDirectedEdges(9, 20);dg.addDirectedEdges(10, 20);dg.addDirectedEdges(15, 20);dg.addDirectedEdges(16, 20);dg.addDirectedEdges(18, 20);dg.addDirectedEdges(19, 20);dg.addDirectedEdges(21, 20);dg.addDirectedEdges(1, 21);dg.addDirectedEdges(7, 21);dg.addDirectedEdges(8, 21);dg.addDirectedEdges(9, 21);dg.addDirectedEdges(10, 21);dg.addDirectedEdges(11, 21);dg.addDirectedEdges(16, 21);dg.addDirectedEdges(19, 21);dg.addDirectedEdges(20, 21);dg.addDirectedEdges(22, 21);dg.addDirectedEdges(1, 22);dg.addDirectedEdges(9, 22);dg.addDirectedEdges(10, 22);dg.addDirectedEdges(11, 22);dg.addDirectedEdges(16, 22);dg.addDirectedEdges(19, 22);dg.addDirectedEdges(21, 22);return dg;}
	public static SteinerDirectedInstance get_dg2() {DirectedGraph dg = get_dg1();SteinerDirectedInstance sdi = new SteinerDirectedInstance(dg);sdi.setRoot(1);sdi.setRequiredNodes(2, 3, 4, 5, 6, 7, 8, 9, 10, 11);sdi.setCost(1, 2, 1000.00);sdi.setCost(3, 2, 74.30);sdi.setCost(12, 2, 139.40);sdi.setCost(13, 2, 237.90);sdi.setCost(14, 2, 285.90);sdi.setCost(15, 2, 397.90);sdi.setCost(16, 2, 673.70);sdi.setCost(1, 3, 1000.00);sdi.setCost(2, 3, 61.10);sdi.setCost(4, 3, 155.70);sdi.setCost(12, 3, 51.00);sdi.setCost(13, 3, 137.50);sdi.setCost(14, 3, 136.00);sdi.setCost(15, 3, 188.70);sdi.setCost(16, 3, 400.20);sdi.setCost(1, 4, 1000.00);sdi.setCost(3, 4, 174.40);sdi.setCost(5, 4, 51.90);sdi.setCost(6, 4, 166.60);sdi.setCost(12, 4, 217.90);sdi.setCost(13, 4, 354.90);sdi.setCost(14, 4, 472.60);sdi.setCost(15, 4, 630.50);sdi.setCost(16, 4, 851.10);sdi.setCost(17, 4, 189.60);sdi.setCost(18, 4, 257.60);sdi.setCost(19, 4, 500.90);sdi.setCost(1, 5, 1000.00);sdi.setCost(4, 5, 44.50);sdi.setCost(6, 5, 70.90);sdi.setCost(12, 5, 134.10);sdi.setCost(13, 5, 244.30);sdi.setCost(14, 5, 262.70);sdi.setCost(15, 5, 326.00);sdi.setCost(16, 5, 506.90);sdi.setCost(17, 5, 98.50);sdi.setCost(18, 5, 118.90);sdi.setCost(19, 5, 270.30);sdi.setCost(1, 6, 1000.00);sdi.setCost(4, 6, 87.60);sdi.setCost(5, 6, 66.20);sdi.setCost(7, 6, 80.20);sdi.setCost(13, 6, 128.80);sdi.setCost(14, 6, 178.40);sdi.setCost(15, 6, 228.20);sdi.setCost(16, 6, 329.10);sdi.setCost(17, 6, 28.70);sdi.setCost(18, 6, 60.30);sdi.setCost(19, 6, 180.60);sdi.setCost(1, 7, 1000.00);sdi.setCost(6, 7, 159.60);sdi.setCost(8, 7, 36.80);sdi.setCost(13, 7, 336.10);sdi.setCost(14, 7, 378.90);sdi.setCost(15, 7, 469.60);sdi.setCost(16, 7, 586.80);sdi.setCost(17, 7, 194.60);sdi.setCost(18, 7, 255.10);sdi.setCost(19, 7, 410.70);sdi.setCost(20, 7, 63.40);sdi.setCost(21, 7, 202.50);sdi.setCost(1, 8, 1000.00);sdi.setCost(7, 8, 50.40);sdi.setCost(9, 8, 171.80);sdi.setCost(14, 8, 472.60);sdi.setCost(15, 8, 470.70);sdi.setCost(16, 8, 733.10);sdi.setCost(17, 8, 156.60);sdi.setCost(18, 8, 229.50);sdi.setCost(19, 8, 514.70);sdi.setCost(20, 8, 46.20);sdi.setCost(21, 8, 199.80);sdi.setCost(1, 9, 1000.00);sdi.setCost(8, 9, 199.60);sdi.setCost(10, 9, 55.20);sdi.setCost(14, 9, 903.00);sdi.setCost(15, 9, 944.20);sdi.setCost(16, 9, 1519.90);sdi.setCost(17, 9, 528.20);sdi.setCost(18, 9, 612.10);sdi.setCost(19, 9, 990.70);sdi.setCost(20, 9, 242.10);sdi.setCost(21, 9, 596.80);sdi.setCost(22, 9, 207.90);sdi.setCost(1, 10, 1000.00);sdi.setCost(9, 10, 44.00);sdi.setCost(15, 10, 696.90);sdi.setCost(16, 10, 750.40);sdi.setCost(18, 10, 382.00);sdi.setCost(19, 10, 565.80);sdi.setCost(20, 10, 151.70);sdi.setCost(21, 10, 314.80);sdi.setCost(22, 10, 89.30);sdi.setCost(1, 11, 1000.00);sdi.setCost(16, 11, 406.60);sdi.setCost(19, 11, 282.00);sdi.setCost(21, 11, 157.00);sdi.setCost(22, 11, 93.90);sdi.setCost(1, 12, 1000.00);sdi.setCost(2, 12, 65.90);sdi.setCost(3, 12, 45.20);sdi.setCost(4, 12, 130.50);sdi.setCost(5, 12, 91.80);sdi.setCost(13, 12, 74.40);sdi.setCost(14, 12, 70.60);sdi.setCost(15, 12, 122.90);sdi.setCost(16, 12, 238.90);sdi.setCost(1, 13, 1000.00);sdi.setCost(2, 13, 132.80);sdi.setCost(3, 13, 76.40);sdi.setCost(4, 13, 135.70);sdi.setCost(5, 13, 157.50);sdi.setCost(6, 13, 100.10);sdi.setCost(7, 13, 193.80);sdi.setCost(12, 13, 51.70);sdi.setCost(14, 13, 36.30);sdi.setCost(15, 13, 63.80);sdi.setCost(16, 13, 134.50);sdi.setCost(1, 14, 1000.00);sdi.setCost(2, 14, 127.20);sdi.setCost(3, 14, 104.50);sdi.setCost(4, 14, 251.30);sdi.setCost(5, 14, 178.10);sdi.setCost(6, 14, 127.00);sdi.setCost(7, 14, 250.90);sdi.setCost(8, 14, 229.20);sdi.setCost(9, 14, 295.30);sdi.setCost(12, 14, 55.90);sdi.setCost(13, 14, 38.10);sdi.setCost(15, 14, 45.30);sdi.setCost(16, 14, 139.30);sdi.setCost(17, 14, 104.10);sdi.setCost(1, 15, 1000.00);sdi.setCost(2, 15, 144.30);sdi.setCost(3, 15, 99.80);sdi.setCost(4, 15, 182.80);sdi.setCost(5, 15, 166.60);sdi.setCost(6, 15, 113.10);sdi.setCost(7, 15, 228.50);sdi.setCost(8, 15, 220.90);sdi.setCost(9, 15, 275.40);sdi.setCost(10, 15, 182.80);sdi.setCost(12, 15, 86.20);sdi.setCost(13, 15, 60.00);sdi.setCost(14, 15, 34.00);sdi.setCost(16, 15, 83.30);sdi.setCost(17, 15, 116.60);sdi.setCost(18, 15, 90.20);sdi.setCost(20, 15, 157.40);sdi.setCost(1, 16, 1000.00);sdi.setCost(2, 16, 84.00);sdi.setCost(3, 16, 105.40);sdi.setCost(4, 16, 142.00);sdi.setCost(5, 16, 129.20);sdi.setCost(6, 16, 93.70);sdi.setCost(7, 16, 135.70);sdi.setCost(8, 16, 103.20);sdi.setCost(9, 16, 179.80);sdi.setCost(10, 16, 125.20);sdi.setCost(11, 16, 106.70);sdi.setCost(12, 16, 86.60);sdi.setCost(13, 16, 62.20);sdi.setCost(14, 16, 65.30);sdi.setCost(15, 16, 43.00);sdi.setCost(17, 16, 93.50);sdi.setCost(18, 16, 102.90);sdi.setCost(19, 16, 59.00);sdi.setCost(20, 16, 98.60);sdi.setCost(21, 16, 86.20);sdi.setCost(22, 16, 124.00);sdi.setCost(1, 17, 1000.00);sdi.setCost(4, 17, 143.20);sdi.setCost(5, 17, 94.60);sdi.setCost(6, 17, 37.40);sdi.setCost(7, 17, 141.70);sdi.setCost(8, 17, 118.60);sdi.setCost(9, 17, 275.70);sdi.setCost(14, 17, 167.20);sdi.setCost(15, 17, 204.40);sdi.setCost(16, 17, 345.10);sdi.setCost(18, 17, 48.70);sdi.setCost(19, 17, 165.90);sdi.setCost(1, 18, 1000.00);sdi.setCost(4, 18, 135.30);sdi.setCost(5, 18, 112.80);sdi.setCost(6, 18, 63.60);sdi.setCost(7, 18, 154.50);sdi.setCost(8, 18, 118.40);sdi.setCost(9, 18, 244.90);sdi.setCost(10, 18, 235.30);sdi.setCost(15, 18, 129.70);sdi.setCost(16, 18, 202.80);sdi.setCost(17, 18, 37.70);sdi.setCost(19, 18, 91.90);sdi.setCost(20, 18, 98.70);sdi.setCost(1, 19, 1000.00);sdi.setCost(4, 19, 99.40);sdi.setCost(5, 19, 93.80);sdi.setCost(6, 19, 85.30);sdi.setCost(7, 19, 105.60);sdi.setCost(8, 19, 128.40);sdi.setCost(9, 19, 169.90);sdi.setCost(10, 19, 144.50);sdi.setCost(11, 19, 129.70);sdi.setCost(16, 19, 71.80);sdi.setCost(17, 19, 62.80);sdi.setCost(18, 19, 56.60);sdi.setCost(20, 19, 99.00);sdi.setCost(21, 19, 70.70);sdi.setCost(22, 19, 118.10);sdi.setCost(1, 20, 1000.00);sdi.setCost(7, 20, 71.30);sdi.setCost(8, 20, 44.70);sdi.setCost(9, 20, 152.70);sdi.setCost(10, 20, 120.70);sdi.setCost(15, 20, 327.40);sdi.setCost(16, 20, 438.00);sdi.setCost(18, 20, 123.50);sdi.setCost(19, 20, 274.10);sdi.setCost(21, 20, 100.40);sdi.setCost(1, 21, 1000.00);sdi.setCost(7, 21, 73.60);sdi.setCost(8, 21, 79.50);sdi.setCost(9, 21, 127.50);sdi.setCost(10, 21, 117.80);sdi.setCost(11, 21, 120.60);sdi.setCost(16, 21, 156.80);sdi.setCost(19, 21, 80.90);sdi.setCost(20, 21, 54.20);sdi.setCost(22, 21, 61.90);sdi.setCost(1, 22, 1000.00);sdi.setCost(9, 22, 77.80);sdi.setCost(10, 22, 54.30);sdi.setCost(11, 22, 73.60);sdi.setCost(16, 22, 268.10);sdi.setCost(19, 22, 178.40);sdi.setCost(21, 22, 91.60);return sdi;}
	public static void select_reuse_pairs() {SteinerDirectedInstance sdi = get_dg2();SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();alg.setInstance(sdi);alg.compute();System.out.println("Returned solution : " + alg.getArborescence());System.out.println("Cost: " + alg.getCost());System.out.println("Running Time: " + alg.getTime() + " ms");String selected = "[";for (Arc a : alg.getArborescence())selected += "("+a.getInput()+", " + a.getOutput() + "),";selected += "]";System.out.println(selected);for (Arc a : alg.getArborescence())sdi.getGraph().setColor(a, Color.RED);new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());}
	
	
	public static void select_reuse_pairs_old() {

		DirectedGraph dg = new DirectedGraph();

		// Add 5 vertices : 1, 2, 3, 4 and 5
		for (int i = 1; i <= 5; i++)
			dg.addVertice(i);

		// Add arcs
		dg.addDirectedEdges(1, 2, 3); // Add an arc (1,2) and an arc (1,3)
		dg.addDirectedEdges(2, 4, 5); // Add an arc (2,4) and an arc (2,5)
		dg.addDirectedEdges(3, 1, 5);
		dg.addDirectedEdges(4, 5);

		SteinerDirectedInstance sdi = new SteinerDirectedInstance(dg);
		sdi.setRoot(1); // Set the node 1 as the root of the instance
		sdi.setRequiredNodes(4, 5); // Set the nodes 4 and 5 as the terminals
		sdi.setCost(3, 5, 1.5); // Set the cost of the arc (3,5) as 5
		sdi.setCost(4, 5, 1.1); // Set the cost of the arc (4,5) as 3
		sdi.setCost(2, 5, 1.1); // Set the cost of the arc (2,5) as 2
		// Every other cost is the default cost: 1

		// Create an algorithm to solve or approximate the instance
		ShPAlgorithm alg = new ShPAlgorithm();
		alg.setInstance(sdi);
		alg.compute();

		// Display the solution, the cost of the solution and the time needed to
		// compute them
		System.out.println("Returned solution : " + alg.getArborescence());
		System.out.println("Cost: " + alg.getCost());
		System.out.println("Running Time: " + alg.getTime() + " ms");
		
		// Get the selected edges and print them
		String selected = "[";
		for (Arc a : alg.getArborescence())
			selected += "("+a.getInput()+", " + a.getOutput() + "),";
		selected += "]";
		System.out.println(selected);
		
		// Display the graph on the screen

		// We display the returned solution in red
		for (Arc a : alg.getArborescence())
		{	
			sdi.getGraph().setColor(a, Color.RED);
		}	
		// We now display the graph and the cost of the edges on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}
	
	

	/**
	 * Create a Steiner instance and run an approximation over it. Finally draw
	 * the instance and the returned solution on the screen.
	 */
	public static void exampleCreateInstanceAndDisplay() {

		DirectedGraph dg = new DirectedGraph();

		// Add 5 vertices : 1, 2, 3, 4 and 5
		for (int i = 1; i <= 5; i++)
			dg.addVertice(i);

		// Add arcs
		dg.addDirectedEdges(1, 2, 3); // Add an arc (1,2) and an arc (1,3)
		dg.addDirectedEdges(2, 4, 5); // Add an arc (2,4) and an arc (2,5)
		dg.addDirectedEdges(3, 1, 5);
		dg.addDirectedEdges(4, 5);

		SteinerDirectedInstance sdi = new SteinerDirectedInstance(dg);
		sdi.setRoot(1); // Set the node 1 as the root of the instance
		sdi.setRequiredNodes(4, 5); // Set the nodes 4 and 5 as the terminals
		sdi.setCost(3, 5, 5); // Set the cost of the arc (3,5) as 5
		sdi.setCost(4, 5, 3); // Set the cost of the arc (4,5) as 3
		sdi.setCost(2, 5, 2); // Set the cost of the arc (2,5) as 2
		// Every other cost is the default cost: 1

		// Create an algorithm to solve or approximate the instance
		ShPAlgorithm alg = new ShPAlgorithm();
		alg.setInstance(sdi);
		alg.compute();

		// Display the solution, the cost of the solution and the time needed to
		// compute them
		System.out.println("Returned solution : " + alg.getArborescence());
		System.out.println("Cost: " + alg.getCost());
		System.out.println("Running Time: " + alg.getTime() + " ms");

		// Display the graph on the screen

		// We display the returned solution in red
		for (Arc a : alg.getArborescence())
			sdi.getGraph().setColor(a, Color.RED);
		// We now display the graph and the cost of the edges on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}

	/**
	 * Create a random Steiner instance and run an approximation over it. Finally draw
	 * the instance and the returned solution on the screen.
	 */
	public static void exampleCreateRandomInstanceAndDisplay() {

		RandomSteinerDirectedGraphGenerator2 gen = new RandomSteinerDirectedGraphGenerator2();
		gen.setNumberOfVerticesLaw(10);
		gen.setNumberOfRequiredVerticesLaw(4);
		gen.setProbabilityOfLinkLaw(new BBernouilliLaw(0.3));
		gen.setCostLaw(1);
		
		SteinerDirectedInstance sdi = gen.generate();
		
		// Create an algorithm to solve or approximate the instance
		ShPAlgorithm alg = new ShPAlgorithm();
		alg.setInstance(sdi);
		alg.compute();

		// Display the solution, the cost of the solution and the time needed to
		// compute them
		System.out.println("Returned solution : " + alg.getArborescence());
		System.out.println("Cost: " + alg.getCost());
		System.out.println("Running Time: " + alg.getTime() + " ms");

		// Display the graph on the screen

		// We display the returned solution in red
		for (Arc a : alg.getArborescence())
			sdi.getGraph().setColor(a, Color.RED);
		// We now display the graph and the cost of the edges on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}


	/*------------------------------------------------------------------------
	 * 
	 * Examples on how to transform an undirected steiner instance into a directed instance.
	 * 
	 *------------------------------------------------------------------------
	 */


	/**
	 * Transform a small undirected instance into a bidirected Steiner instance.
	 */
	public static void exampleTransformUndirectedIntoBidirected(){
		UndirectedGraph ug = new UndirectedGraph();

		// Add 5 vertices : 1, 2, 3, 4 and 5
		for (int i = 1; i <= 5; i++)
			ug.addVertice(i);

		// Add edges
		ug.addUndirectedEdges(1, 2, 3); // Add an edge (1,2) and an edge (1,3)
		ug.addUndirectedEdges(2, 3, 4, 5); // Add edges (2,3) (2,4) (2,5)

		SteinerUndirectedInstance sui = new SteinerUndirectedInstance(ug);
		sui.setRequiredNodes(1, 4, 5); // Set the nodes 1, 4 and 5 as the terminals
		sui.setCost(1, 3, 2); // Set the cost of (1,3) to 2
		// Every other cost is the default cost: 1

		// Transformation
		SteinerDirectedInstance sdi = SteinerDirectedInstance.getSymetrizedGraphFromUndirectedInstance(sui);

		// We now display the graph and the cost of the arcs on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}



	/**
	 * Transform a small undirected instance into an directed acyclic Steiner instance.
	 */
	public static void exampleTransformUndirectedIntoAcyclic(){
		UndirectedGraph ug = new UndirectedGraph();

		// Add 5 vertices : 1, 2, 3, 4 and 5
		for (int i = 1; i <= 5; i++)
			ug.addVertice(i);

		// Add edges
		ug.addUndirectedEdges(1, 2, 3); // Add an edge (1,2) and an edge (1,3)
		ug.addUndirectedEdges(2, 3, 4, 5); // Add edges (2,3) (2,4) (2,5)

		SteinerUndirectedInstance sui = new SteinerUndirectedInstance(ug);
		sui.setRequiredNodes(1, 4, 5); // Set the nodes 1, 4 and 5 as the terminals
		sui.setCost(1, 3, 2); // Set the cost of (1,3) to 2
		// Every other cost is the default cost: 1

		// We describe the optimal solution. This could have been done
		// using an exact algorithm to solve Steiner
		HashSet<Arc> arborescence = new HashSet<Arc>();
		arborescence.add(new Arc(1,2,false));
		arborescence.add(new Arc(2,4,false));
		arborescence.add(new Arc(2,5,false));


		// Transformation
		SteinerDirectedInstance sdi = SteinerDirectedInstance.getAcyclicGraphFromUndirectedInstance(sui, arborescence);

		// We now display the graph and the cost of the arcs on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}


	/**
	 * Transform a small undirected instance into a directed strongly connected Steiner instance.
	 */
	public static void exampleTransformUndirectedIntoStronglyConnected(){
		UndirectedGraph ug = new UndirectedGraph();

		// Add 5 vertices : 1, 2, 3, 4 and 5
		for (int i = 1; i <= 5; i++)
			ug.addVertice(i);

		// Add edges
		ug.addUndirectedEdges(1, 2, 3); // Add an edge (1,2) and an edge (1,3)
		ug.addUndirectedEdges(2, 3, 4, 5); // Add edges (2,3) (2,4) (2,5)

		SteinerUndirectedInstance sui = new SteinerUndirectedInstance(ug);
		sui.setRequiredNodes(1, 4, 5); // Set the nodes 1, 4 and 5 as the terminals
		sui.setCost(1, 3, 2); // Set the cost of (1,3) to 2
		// Every other cost is the default cost: 1

		// We describe the optimal solution. This could have been done
		// using an exact algorithm to solve Steiner
		HashSet<Arc> arborescence = new HashSet<Arc>();
		arborescence.add(new Arc(1,2,false));
		arborescence.add(new Arc(2,4,false));
		arborescence.add(new Arc(2,5,false));


		// Transformation
		SteinerDirectedInstance sdi = SteinerDirectedInstance.getRandomGraphStronglyConnectedFromUndirectedInstance(sui, arborescence);

		// We now display the graph and the cost of the arcs on the screen
		new EnergyAnalogyGraphDrawer(sdi.getGraph(), sdi.getCosts());
	}

	/*-------------------------------------------------------------------------------
	 * 
	 * 
	 * Example on how to generate the benchmarks of directed instances from a benchmark of undirected instances.
	 * 
	 * We assume:
	 * - that all the undirected instances of the SteinLib Benchmark were downloaded at http://steinlib.zib.de/ with the STP format and
	 * stored in a main SteinLib directory. For example, here, we assume this directory is "SteinLib/".
	 * - that each instance category has its own subdirectory. For example here, we test the algorithms over the
	 * category "B" from the "Sparse Complete Other" main category of instances (see http://http://steinlib.zib.de/testset.php).
	 * We downloaded the B instances in a subFolder named "SteinLib/B/"
	 * - that the results of category "B"  are in the folder "SteinLib/Results/B.results".
	 * 
	 * (Notice one can currently find all the results for all the categories in the directory "/SteinLibOptimalSolutions/")
	 * 
	 * -------------------------------------------------------------------------------
	 */

	public static void exampleCreateBidirectedInstances() {
		String steinLibMainDir = "SteinLib/";
		String steinLibSubDir = "B/";
		String steinLibTargetMainDir = "SteinLibBidir/";

		createBidirectedInstances(steinLibMainDir, steinLibSubDir,
				steinLibTargetMainDir);
	}

	public static void exampleCreateAcyclicInstances() {
		String steinLibMainDir = "SteinLib/";
		String steinLibSubDir = "B/";
		String steinLibTargetMainDir = "SteinLibAcyclic/";

		createAcyclicInstances(steinLibMainDir, steinLibSubDir,
				steinLibTargetMainDir);
	}

	public static void exampleCreateSronglyConnectedInstances() {
		String steinLibMainDir = "SteinLib/";
		String steinLibSubDir = "B/";
		String steinLibTargetMainDir = "SteinLibStrongly/";

		createStronglyConnectedInstances(steinLibMainDir, steinLibSubDir,
				steinLibTargetMainDir);
	}

	/**
	 * From undirected instances in "steinLibMainDir/steinLibSubDir", create
	 * bidirected instances, and store them with the STP format in the directory
	 * "steinLibTargetMainDir/steinLibSubDir".
	 * <p>
	 * If the original instance name is ABC01.stp, the name of the computed
	 * instance is ABC01bd.stp
	 * 
	 * 
	 * @param steinLibMainDir
	 * @param steinLibSubDir
	 * @param steinLibTargetMainDir
	 */
	public static void createBidirectedInstances(String steinLibMainDir,
			String steinLibSubDir, String steinLibTargetMainDir) {
		File f = new File(steinLibMainDir + steinLibSubDir);
		String name = f.listFiles()[0].getName();
		name = name.substring(0, name.length() - 4); 
		SteinLibInstancesGroups slig = SteinLibInstancesGroups.getGroup(name);

		// File containing for each instance the  optimal solution cost
		String resultFilePath = slig.getResultFileName();

		// Make the target directories if they do not exists.
		File mkdir = new File(steinLibTargetMainDir+steinLibSubDir);
		mkdir.mkdirs();
		mkdir = new File(steinLibTargetMainDir+"Results");
		mkdir.mkdirs();

		STPUndirectedGenerator gen = new STPUndirectedGenerator(steinLibMainDir
				+ steinLibSubDir, steinLibMainDir + resultFilePath);

		FileManager writeOptimalSolutionsValues = new FileManager();
		writeOptimalSolutionsValues.openErase(steinLibTargetMainDir+resultFilePath);
		for (int i = 0; i < gen.getNumberOfInstances(); i++) {
			SteinerUndirectedInstance sui = gen.generate();
			SteinerDirectedInstance sdi = SteinerDirectedInstance
					.getSymetrizedGraphFromUndirectedInstance(sui);

			String instanceName = sui.getGraph().getParam(
					STPGenerator.OUTPUT_NAME_PARAM_NAME)
					+ "bd";

			Integer instanceOptimumValue = sui.getGraph().getParamInteger(
					STPGenerator.OUTPUT_OPTIMUM_VALUE_PARAM_NAME);

			writeOptimalSolutionsValues.writeln(instanceName+" "+instanceOptimumValue);

			STPTranslator.translateSteinerGraph(
					sdi,
					steinLibTargetMainDir
					+ steinLibSubDir
					+ instanceName+".stp");

		}
		writeOptimalSolutionsValues.closeWrite();
	}

	/**
	 * From undirected instances in "steinLibMainDir/steinLibSubDir", create
	 * acyclic instances, and store them with the STP format in the directory
	 * "steinLibTargetMainDir/steinLibSubDir".
	 * <p>
	 * If the original instance name is ABC01.stp, the name of the computed
	 * instance is ABC01ac.stp
	 * 
	 * @param steinLibMainDir
	 * @param steinLibSubDir
	 * @param steinLibTargetMainDir
	 */
	public static void createAcyclicInstances(String steinLibMainDir,
			String steinLibSubDir, String steinLibTargetMainDir) {
		File f = new File(steinLibMainDir + steinLibSubDir);
		String name = f.listFiles()[0].getName();
		name = name.substring(0, name.length() - 4);
		SteinLibInstancesGroups slig = SteinLibInstancesGroups.getGroup(name);

		// File containing for each instance the  optimal solution cost
		String resultFilePath = slig.getResultFileName();


		// Make the target directories if they do not exists.
		File mkdir = new File(steinLibTargetMainDir+steinLibSubDir);
		mkdir.mkdirs();
		mkdir = new File(steinLibTargetMainDir+"Results");
		mkdir.mkdirs();

		STPUndirectedGenerator gen = new STPUndirectedGenerator(steinLibMainDir
				+ steinLibSubDir, steinLibMainDir + resultFilePath);

		FileManager writeOptimalSolutionsValues = new FileManager();
		writeOptimalSolutionsValues.openErase(steinLibTargetMainDir+resultFilePath);
		for (int i = 0; i < gen.getNumberOfInstances(); i++) {
			SteinerUndirectedInstance sui = gen.generate();

			@SuppressWarnings("unchecked")
			HashSet<Arc> arborescence = (HashSet<Arc>) sui.getGraph().getParam(
					STPGenerator.OUTPUT_OPTIMUM_PARAM_NAME);
			if(arborescence == null)
				continue;

			SteinerDirectedInstance sdi = SteinerDirectedInstance
					.getAcyclicGraphFromUndirectedInstance(
							sui, arborescence);

			String instanceName = sui.getGraph().getParam(
					STPGenerator.OUTPUT_NAME_PARAM_NAME)
					+ "ac";

			Integer instanceOptimumValue = sui.getGraph().getParamInteger(
					STPGenerator.OUTPUT_OPTIMUM_VALUE_PARAM_NAME);

			writeOptimalSolutionsValues.writeln(instanceName+" "+instanceOptimumValue);

			STPTranslator.translateSteinerGraph(
					sdi,
					steinLibTargetMainDir
					+ steinLibSubDir
					+ instanceName+".stp");
		}
		writeOptimalSolutionsValues.closeWrite();
	}

	/**
	 * From undirected instances in "steinLibMainDir/steinLibSubDir", create
	 * strongly connected instances, and store them with the STP format in the
	 * directory "steinLibTargetMainDir/steinLibSubDir".
	 * <p>
	 * If the original instance name is ABC01.stp, the name of the computed
	 * instance is ABC01st.stp
	 * 
	 * @param steinLibMainDir
	 * @param steinLibSubDir
	 * @param steinLibTargetMainDir
	 */
	public static void createStronglyConnectedInstances(
			String steinLibMainDir, String steinLibSubDir,
			String steinLibTargetMainDir) {
		File f = new File(steinLibMainDir + steinLibSubDir);
		String name = f.listFiles()[0].getName();
		name = name.substring(0, name.length() - 4);
		SteinLibInstancesGroups slig = SteinLibInstancesGroups.getGroup(name);

		// File containing for each instance the  optimal solution cost
		String resultFilePath = slig.getResultFileName();


		// Make the target directories if they do not exists.
		File mkdir = new File(steinLibTargetMainDir+steinLibSubDir);
		mkdir.mkdirs();
		mkdir = new File(steinLibTargetMainDir+"Results");
		mkdir.mkdirs();

		STPUndirectedGenerator gen = new STPUndirectedGenerator(steinLibMainDir
				+ steinLibSubDir, steinLibMainDir + resultFilePath);

		FileManager writeOptimalSolutionsValues = new FileManager();
		writeOptimalSolutionsValues.openErase(steinLibTargetMainDir+resultFilePath);
		for (int i = 0; i < gen.getNumberOfInstances(); i++) {
			SteinerUndirectedInstance sui = gen.generate();
			@SuppressWarnings("unchecked")
			HashSet<Arc> arborescence = (HashSet<Arc>) sui.getGraph().getParam(
					STPGenerator.OUTPUT_OPTIMUM_PARAM_NAME);
			if(arborescence == null)
				continue;

			SteinerDirectedInstance sdi = SteinerDirectedInstance
					.getRandomGraphStronglyConnectedFromUndirectedInstance(
							sui, arborescence);

			String instanceName = sui.getGraph().getParam(
					STPGenerator.OUTPUT_NAME_PARAM_NAME)
					+ "st";

			Integer instanceOptimumValue = sui.getGraph().getParamInteger(
					STPGenerator.OUTPUT_OPTIMUM_VALUE_PARAM_NAME);

			writeOptimalSolutionsValues.writeln(instanceName+" "+instanceOptimumValue);

			STPTranslator.translateSteinerGraph(
					sdi,
					steinLibTargetMainDir
					+ steinLibSubDir
					+ instanceName+".stp");
		}
		writeOptimalSolutionsValues.closeWrite();
	}

	/*-------------------------------------------------------------------------------
	 * 
	 * 
	 * Example on how to test an algorithm.
	 * The directed instances test must have already been generated using 
	 * the createBidirectedInstances, createAcyclicInstances, and createStronglyConnectedInstances methods.
	 * 
	 * -------------------------------------------------------------------------------
	 */

	/**
	 * This example lauch the evaluation of one algorithm over the instances in
	 * the B category transformed into bidirected instances.
	 * 
	 * We assume those instances were put in the "SteinLibBidir/B/" directory.
	 */
	public static void exampleLaunchBidirTest() {
		// We test that algorithm
		SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();
		int nbInstancesIgnored = 0; // We do not ignore any instance

		// The directory containing all the SteinLib category folders
		// and the "results" folder as explaine previously.
		String steinLibMainDir = "SteinLibBidir/";

		String steinLibSubDir = "B/"; // Inside the main dir, the subdirectory
		// containing the instance is B/

		testAlgorithm(steinLibMainDir, steinLibSubDir, nbInstancesIgnored, alg);
	}

	/**
	 * This example lauch the evaluation of one algorithm over the instances in
	 * the B category transformed into acyclic instances.
	 * 
	 * We assume those instances were put in the "SteinLibAcyclic/B/" directory.
	 */
	public static void exampleLaunchAcyclicTest() {
		// We test that algorithm
		SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();
		int nbInstancesIgnored = 0; // We do not ignore any instance

		// The directory containing all the SteinLib category folders
		// and the "results" folder as explaine previously.
		String steinLibMainDir = "SteinLibAcyclic/";

		String steinLibSubDir = "B/"; // Inside the main dir, the subdirectory
		// containing the instance is B/

		testAlgorithm(steinLibMainDir, steinLibSubDir, nbInstancesIgnored, alg);
	}

	/**
	 * This example lauch the evaluation of one algorithm over the instances in
	 * the B category transformed into strongly connected instances.
	 * 
	 * We assume those instances were put in the "SteinLibStrongly/B/" directory.
	 */
	public static void exampleLaunchStronglyTest() {
		// We test that algorithm
		SteinerArborescenceApproximationAlgorithm alg = new GFLACAlgorithm();
		int nbInstancesIgnored = 0; // We do not ignore any instance

		// The directory containing all the SteinLib category folders
		// and the "results" folder as explaine previously.
		String steinLibMainDir = "SteinLibStrongly/";

		String steinLibSubDir = "B/"; // Inside the main dir, the subdirectory
		// containing the instance is B/

		testAlgorithm(steinLibMainDir, steinLibSubDir, nbInstancesIgnored, alg);
	}

	/**
	 * Test the algorithm alg over all instances in the directory
	 * steinLibDir/steinLibSubDir/ Ignore the nbInstancesIgnored first instances
	 * <p>
	 * Instances in the directory steinLibDir/steinLibSubDir/ must use the STP
	 * format from the SteinLib benchmark.
	 * 
	 * @param steinLibMainDir
	 * @param steinLibSubDir
	 * @param nbInstancesIgnored
	 * @param alg
	 */
	public static void testAlgorithm(String steinLibMainDir,
			String steinLibSubDir, int nbInstancesIgnored,
			SteinerArborescenceApproximationAlgorithm alg) {

		// Description
		System.out.println("# Name OptimalCost NbNodes NbArcs NbTerminals MaximumArcCost AlgorithmAnswer AlgorithRunningTime");

		File f = new File(steinLibMainDir + steinLibSubDir);
		String name = f.listFiles()[0].getName();
		name = name.substring(0, name.length() - 6);
		SteinLibInstancesGroups slig = SteinLibInstancesGroups.getGroup(name);
		String path2 = slig.getResultFileName(); // File containing for each
		// instance the optimal
		// solution cost
		STPDirectedGenerator gen = new STPDirectedGenerator(steinLibMainDir
				+ steinLibSubDir, steinLibMainDir + path2);

		gen.incrIndex(nbInstancesIgnored);
		alg.setCheckFeasibility(false);
		for (int i = nbInstancesIgnored; i < gen.getNumberOfInstances(); i++) {
			SteinerDirectedInstance sdi = gen.generate();

			System.out.print(sdi.getGraph().getParam(
					STPDirectedGenerator.OUTPUT_NAME_PARAM_NAME)
					+ " "); // Show the name of the instance
			System.out.print(sdi.getGraph().getParam(
					STPDirectedGenerator.OUTPUT_OPTIMUM_VALUE_PARAM_NAME)
					+ " "); // Show the optimal cost of the instance
			System.out.print(sdi.getGraph().getNumberOfVertices() + " "
					+ sdi.getGraph().getNumberOfEdges() + " "
					+ sdi.getNumberOfRequiredVertices() + " " + sdi.maxCost()
					+ " "); // Show some informations of the instance

			alg.setInstance(sdi);
			alg.compute(); // Run the algorithm over the instance
			// Show the results and the running time
			System.out.print(alg.getCost() + " " + alg.getTime() + " ");
			System.out.println();
		}
	}

	public static void testAlgorithmPersoInstances(String dir, int nbInstancesIgnored,
			SteinerArborescenceApproximationAlgorithm alg) {

		// Description
		System.out.println("# Name OptimalCost NbNodes NbArcs NbTerminals MaximumArcCost AlgorithmAnswer AlgorithRunningTime");

		File f = new File(dir);
		STPDirectedGenerator gen = new STPDirectedGenerator(dir, null);

		gen.incrIndex(nbInstancesIgnored);
		alg.setCheckFeasibility(false);
		for (int i = nbInstancesIgnored; i < gen.getNumberOfInstances(); i++) {
			SteinerDirectedInstance sdi = gen.generate();
			if (sdi == null)
				continue;

			System.out.print(sdi.getGraph().getParam(
					STPDirectedGenerator.OUTPUT_NAME_PARAM_NAME)
					+ " "); // Show the name of the instance
			System.out.print(sdi.getGraph().getNumberOfVertices() + " "
					+ sdi.getGraph().getNumberOfEdges() + " "
					+ sdi.getNumberOfRequiredVertices() + " " + sdi.maxCost()
					+ " "); // Show some informations of the instance

			alg.setInstance(sdi);
			alg.compute(); // Run the algorithm over the instance
			// Show the results and the running time
			System.out.print(alg.getCost() + " " + alg.getTime() + " ");
			System.out.println();
		}
	}
}
