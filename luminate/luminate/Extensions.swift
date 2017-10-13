//
//  Extensions.swift
//  luminate
//
//  Created by Prayash Thapa on 10/7/17.
//  Copyright © 2017 com.reality.af. All rights reserved.
//
import UIKit
import Foundation

// MARK: Float Extension

public extension Float {
    
    public static var random: Float {
        return Float(arc4random()) / 0xFFFFFFFF
    }
    
    public static func random(min: Float, max: Float) -> Float {
        return Float.random * (max - min) + min
    }
}
extension UIView {
    func fadeIn() {
        UIView.animate(withDuration: 1.0, delay: 0.0, options: UIViewAnimationOptions.curveEaseIn, animations: {
            self.alpha = 1.0
        }, completion: nil)
    }
    
    func fadeOut() {
        UIView.animate(withDuration: 1.0, delay: 0.0, options: UIViewAnimationOptions.curveEaseOut, animations: {
            self.alpha = 0.0
        }, completion: nil)
    }
}
