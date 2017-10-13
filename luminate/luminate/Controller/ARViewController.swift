//
//  ViewController.swift
//  luminate
//
//  Created by Jesse Litton on 10/6/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import UIKit
import ARKit
import SceneKit
import CoreLocation
import Firebase
import FirebaseDatabase

class ARViewController: UIViewController, ARSCNViewDelegate, SCNSceneRendererDelegate, UIGestureRecognizerDelegate {
    
    // MARK: - Properties
    
    var data = [UUID: Lumen]()
    var locationManager = CLLocationManager()
    var currentLocation: String!
    
    var isGiving: Bool!
    
    var ref: DatabaseReference = Database.database().reference()
    var castLantern: Lumen!
    let sceneView =  ARSCNView()
    let detailView = DetailView()
    
    let sprites = [UIImage(named: "sprite1"),
                   UIImage(named: "sprite2"),
                   UIImage(named: "sprite3")]
    
    // MARK: - View Controller Lifecycle
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        setupScene()
        setupRecognizers()
        setupSubviews()
        getLocation()
    }
    
    func getLocation() {
        locationManager.requestWhenInUseAuthorization()
        if (CLLocationManager.authorizationStatus() == CLAuthorizationStatus.authorizedWhenInUse ||
            CLLocationManager.authorizationStatus() == CLAuthorizationStatus.authorizedAlways){
            print(locationManager.location!)
            
            let geocoder = CLGeocoder()
            geocoder.reverseGeocodeLocation(locationManager.location!) { (placemarks, error) in
                self.currentLocation = placemarks?.first?.locality
                print(self.currentLocation)
                
                if self.isGiving == nil {
                    self.bindDataObserver()
                } else {
                    self.setupLumenCast()
                }
                
//                if self.isGiving {
//                    self.setupLumenCast()
//                } else {
//                    self.bindDataObserver()
//                }
            }
        }
    }
    
    override func viewWillAppear(_ animated: Bool) {
        super.viewWillAppear(animated)
        setupSession()
        
        self.navigationController?.setNavigationBarHidden(true, animated: animated)
    }
    
    override func viewWillDisappear(_ animated: Bool) {
        super.viewWillDisappear(animated)
        
        // Pause the view's session
        sceneView.session.pause()
    }
    
    // MARK: - Setup Methods
    
    func setupScene() {
        view.addSubview(sceneView)
        sceneView.anchor(view.topAnchor, left: view.leftAnchor, bottom: view.bottomAnchor, right: view.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 0)
        
        // Setup the ARSCNViewDelegate - this gives us callbacks to handle new
        // geometry creation
        self.sceneView.delegate = self
        
        // Adds default lighting to scene
        self.sceneView.autoenablesDefaultLighting = true
        
        // Show statistics such as fps and timing information
        self.sceneView.showsStatistics = true
        
        // Show debug information for feature tracking
        self.sceneView.debugOptions = [ARSCNDebugOptions.showFeaturePoints] // ARSCNDebugOptions.showWorldOrigin
        
        // Create a new scene by loading it from scene assets
        let scene = SCNScene()

        // Set the scene to the view
        self.sceneView.scene = scene
        self.sceneView.isPlaying = true
        self.sceneView.scene.physicsWorld.gravity = SCNVector3Make(0.0, 0.0, 0.0)
    }
    
    func setupSession() {
        // Create a session configuration
        let configuration = ARWorldTrackingConfiguration()
        
        // Specify that we want to track horizontal planes. Setting this will cause
        // the ARSCNViewDelegate methods to be called when planes are detected!
        configuration.planeDetection = .horizontal
        
        // Run the view's session
        sceneView.session.run(configuration)
    }
    
    func setupRecognizers() {
        // Single tap to examine the details about a donation orb
        let tapGestureRecognizer = UITapGestureRecognizer(target: self, action: #selector(handleTapFrom))
        tapGestureRecognizer.numberOfTapsRequired = 1
        self.sceneView.addGestureRecognizer(tapGestureRecognizer)
        self.view.addGestureRecognizer(tapGestureRecognizer)
        
        let swipeUpGesture = UISwipeGestureRecognizer(target: self, action: #selector(handleSwipeUp))
        swipeUpGesture.direction = .up
        self.sceneView.addGestureRecognizer(swipeUpGesture)
    }
    
    func setupSubviews() {
        self.view.addSubview(detailView)
//        self.view.addSubview(goHomeButtonView)
        hideDetails()
        detailView.anchor(self.view.topAnchor, left: self.view.leftAnchor, bottom: self.view.bottomAnchor, right: self.view.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 0)
//        goHomeButtonView.anchor(self.view.topAnchor, left: self.view.leftAnchor, bottom: self.view.bottomAnchor, right: self.view.rightAnchor, topConstant: 0, leftConstant: 0, bottomConstant: 0, rightConstant: 0, widthConstant: 0, heightConstant: 0)
    }
    
    func bindDataObserver() {
        self.ref.child(self.currentLocation).observe(.value) { (snapshot) in
            guard let value = snapshot.value as? NSDictionary else { return }
            
            // print(value!["-Kvw_J0XG04UV4fiGMWj"]!)
            
            for (id, object) in value {
                let node = self.sceneView.scene.rootNode.childNode(withName: id as! String, recursively: true)
                
                if node == nil {
                    self.generateLumen(id: id as! String)
                }
            }
        }
    }
    
    // MARK: - Tap Handling Callbacks
    
    @objc func handleTapFrom(recognizer: UITapGestureRecognizer) {
        let result = self.sceneView.hitTest(recognizer.location(in: self.sceneView), options: [SCNHitTestOption.sortResults : true])
        
    
        if !result.isEmpty {
            displayDetails()
        } else {
            hideDetails()
        }
    }
    
    @objc func handleSwipeUp(recognizer: UITapGestureRecognizer) {
        print("Casting lumen!")
        
        self.ref.child(self.currentLocation).childByAutoId().setValue([
            "username": "prayash",
            "donation": "$20",
            "message": "One love."
        ])
        
        castLantern.moveUpAndDisappear()
    }
    
    func displayDetails() {
        detailView.textView.text = "Basic sanitation is not beyond reach."
        detailView.nameView.text = "ENRICO F."
        detailView.isHidden = false
    }
    
    func hideDetails() {
        detailView.isHidden = true
    }
    
    func setupLumenCast() {
        print("Preparing Lumen for casting")
        castLantern = Lumen(
            id: "cast1",
            position: SCNVector3Make(
                0,
                0.025,
                -0.75
            ),
            size: CGFloat(0.25)
        )
        self.sceneView.scene.rootNode.addChildNode(castLantern)
    }
    
    func generateLumen(id: String) {
        print("Generating lumen... " + id)
        
        let lantern = Lumen(
            id: id,
            position: SCNVector3Make(
                Float.random(min: -1.3, max: 0.75),
                Float.random(min: -1.3, max: 0.75),
                Float.random(min: -2.5, max: 0.0)
            ),
            size: CGFloat(0.125)
        )
        self.sceneView.scene.rootNode.addChildNode(lantern)
        
//        var angle: Float = 0.0
//        let angleInc: Float = Float.pi / Float(sprites.count)
//
//        for i in 0 ..< sprites.count {
//            let radius: Float = Float.random(min: 1.0, max: 2.5)
//            let phi: Float = Float.random(min: 0.5, max: 5)
//            let theta: Float = Float.random(min: 0.5, max: 5)
//
//            let size = Float.random(min: 0.125, max: 0.15)
//            let parentOrb = SCNPlane(width: CGFloat(size), height: CGFloat(size))
//            let billboard = SCNBillboardConstraint()
//            billboard.freeAxes = SCNBillboardAxis.all
//
//            node = SCNNode(geometry: parentOrb)
//            node.constraints = [billboard]
//            node.name = "HELLO Paul"
//            let img = sprites[0]
//            parentOrb.firstMaterial?.diffuse.contents = img
//
//            let displacement: Float = 5.0
//            let up = SCNAction.moveBy(x: 0.0, y: CGFloat(displacement), z: 0.0, duration: 5.0)
//            let down = SCNAction.moveBy(x: 0.0, y: CGFloat(displacement), z: 0.0, duration: 5.0)
//
//            let oscillation = SCNAction.repeatForever(SCNAction.sequence([up, down]))
//            let rotation = SCNAction.repeatForever(SCNAction.rotateBy(x: 1, y: 1, z: 1, duration: 10))
//            let actions = SCNAction.group([oscillation, rotation])
//
//            node.runAction(actions)
//            node.scale = SCNVector3Make(0.5, 0.5, 0.5)
//            node.position = SCNVector3Make(
//                Float.random(min: 0.1, max: 0.2),
//                Float.random(min: 0.1, max: 0.2),
//                -1.5
//            )
//
//            self.sceneView.scene.rootNode.addChildNode(node)
//            angle += angleInc
//        }
    }
    
    // MARK: - Renderer Delegate Methods
    
    func renderer(_ renderer: SCNSceneRenderer, updateAtTime time: TimeInterval) {
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didAdd node: SCNNode, for anchor: ARAnchor) {
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didUpdate node: SCNNode, for anchor: ARAnchor) {
        
    }
    
    func renderer(_ renderer: SCNSceneRenderer, didRemove node: SCNNode, for anchor: ARAnchor) {
        
    }

    // MARK: - ARSCNViewDelegate
    
/*
    // Override to create and configure nodes for anchors added to the view's session.
    func renderer(_ renderer: SCNSceneRenderer, nodeFor anchor: ARAnchor) -> SCNNode? {
        let node = SCNNode()
     
        return node
    }
*/
    
    func session(_ session: ARSession, didFailWithError error: Error) {
        // Present an error message to the user
        resetTracking()
    }
    
    func sessionWasInterrupted(_ session: ARSession) {
        // Inform the user that the session has been interrupted, for example, by presenting an overlay
        resetTracking()
    }
    
    func sessionInterruptionEnded(_ session: ARSession) {
        // Reset tracking and/or remove existing anchors if consistent tracking is required
        resetTracking()
    }
    
    private func resetTracking() {
        let configuration = ARWorldTrackingConfiguration()
        configuration.planeDetection = .horizontal
        sceneView.session.run(configuration, options: [.resetTracking, .removeExistingAnchors])
    }
}
