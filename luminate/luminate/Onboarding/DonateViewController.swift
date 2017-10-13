//
//  DonateViewController.swift
//  luminate
//
//  Created by Prayash Thapa on 10/8/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//

import UIKit

class DonateViewController: UIViewController {
    @IBOutlet weak var castButton: UIButton!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    @IBAction func castButtonDidTouch(_ sender: Any) {
        
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if let vc = segue.destination as? ARViewController {
            vc.isGiving = true
        }
    }
}
